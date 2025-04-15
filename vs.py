import os
import uuid
from flask import Flask, request, send_file
from werkzeug.utils import secure_filename
import ffmpeg
import whisper
from transformers import pipeline
from sqlalchemy import create_engine, Column, Integer, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from flask import Flask, request, send_file, send_from_directory, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__, static_folder="frontend/build")
CORS(app) 

UPLOAD_FOLDER = 'uploads'
SUMMARY_FOLDER = 'summaries'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SUMMARY_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    if os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve(path):
    if path != "" and os.path.exists(f"build/{path}"):
        return send_from_directory("build", path)
    else:
        return send_from_directory("build", "index.html")
    
whisper_model = whisper.load_model("base")

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

Base = declarative_base()
class Feedback(Base):
    __tablename__ = 'feedback'
    id = Column(Integer, primary_key=True)
    summary = Column(Text)
    quiz = Column(Text)
    feedback = Column(Text)

class Transcript(Base):
    __tablename__ = 'transcripts'
    id = Column(Integer, primary_key=True)
    video_filename = Column(Text)
    transcript_text = Column(Text)
    summary_text = Column(Text)

db_engine = create_engine('sqlite:///feedback.db')
Base.metadata.create_all(db_engine)
Session = sessionmaker(bind=db_engine)

def extract_audio(video_path, audio_path):
    ffmpeg.input(video_path).output(audio_path).run(overwrite_output=True)

def transcribe_audio(audio_path):
    result = whisper_model.transcribe(audio_path)
    return result['text']

def split_text(text, max_tokens=800):
    sentences = text.split('. ')
    chunks = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) < max_tokens:
            current += sentence + ". "
        else:
            chunks.append(current.strip())
            current = sentence + ". "
    if current:
        chunks.append(current.strip())
    return chunks

def summarize_long_text(text):
    import requests

    TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
    TOGETHER_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"

    prompt = f"""
    You are an intelligent assistant. Summarize the following transcript:
    Transcript:
    \"\"\"
    {text}
    \"\"\"

    Respond with direct summary
    """

    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": TOGETHER_MODEL,
        "prompt": prompt.strip(),
        "max_tokens": 1024,
        "temperature": 0.7
    }

    try:
        response = requests.post("https://api.together.xyz/inference", json=payload, headers=headers)
        result = response.json()

        if 'choices' in result and result['choices']:
            text = result['choices'][0]['text']
            return text
        else:
            return "Summary generation failed: No valid response."

    except Exception as e:
        return f"Summary generation failed due to error: {str(e)}"

def generate_summary_and_quiz(text, level):
    import requests

    TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
    TOGETHER_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"

    level_prompt = {
        "easy": "Generate simple quiz questions suitable for beginners.",
        "medium": "Generate moderately difficult quiz questions.",
        "hard": "Generate challenging quiz questions requiring deeper understanding."
    }
    summary = summarize_long_text(text)
    quiz_prompt = f"""
    Based on the following transcript and summary, generate a quiz with 3 different types of questions:
    1. Multiple Choice (4 options + correct answer)
    2. True or False (with correct answer)
    3. Fill in the Blanks (with answer)


    {level_prompt.get(level, '')}
    
    Transcript:
    {text}

    Summary:
    {summary}
    
    provide quiz only 
    
    use only this Format:
    Question Type: <Type>
    Question: <Text>
    Options: <A, B, C, D> (for MCQ only)
    Answer: <Answer>
    """

    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": TOGETHER_MODEL,
        "prompt": quiz_prompt.strip(),
        "max_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.9
    }

    try:
        response = requests.post("https://api.together.xyz/inference", json=payload, headers=headers)
        result = response.json()
        if 'choices' in result and len(result['choices']) > 0:
            # Clean quiz text to start from first "Question Type:"
            import re
            match = re.search(r"(Question Type:.*?)$", result['choices'][0]['text'].strip(), re.DOTALL)
            if match:
                quiz = match.group(1).strip()
        else:
            quiz = "Quiz generation failed: Incomplete response from language model."

    except Exception as e:
        print("Error during quiz generation:", e)
        quiz = f"Quiz generation failed due to error: {str(e)}"

    return summary, quiz


@app.route('/process', methods=['POST'])
def process():
    file = request.files['video_file']
    level = request.form.get('level', 'medium')
    filename = secure_filename(file.filename)
    uid = str(uuid.uuid4())
    video_path = os.path.join(UPLOAD_FOLDER, f"{uid}_{filename}")
    audio_path = os.path.join(UPLOAD_FOLDER, f"{uid}_audio.wav")
    file.save(video_path)

    extract_audio(video_path, audio_path)
    transcript = transcribe_audio(audio_path)
    summary, quiz = generate_summary_and_quiz(transcript, level)

    summary_file = os.path.join(SUMMARY_FOLDER, f"{uid}_summary.txt")
    quiz_file = os.path.join(SUMMARY_FOLDER, f"{uid}_quiz.txt")
    transcript_file = os.path.join(SUMMARY_FOLDER, f"{uid}_transcript.txt")
    with open(summary_file, 'w') as f: f.write(summary)
    with open(quiz_file, 'w') as f: f.write(quiz)
    with open(transcript_file, 'w') as f: f.write(transcript)

    session = Session()
    session.add(Transcript(video_filename=filename, transcript_text=transcript, summary_text=summary))
    session.commit()
    session.close()

    if quiz.startswith("Quiz generation failed"):
        return jsonify({"error": quiz}), 500
    return jsonify({
        "summary": summary,
        "quiz": quiz,
        "uid": uid
    })


@app.route('/download/<uid>/<filetype>')
def download(uid, filetype):
    path = os.path.join(SUMMARY_FOLDER, f"{uid}_{filetype}.txt")
    return send_file(path, as_attachment=True)

@app.route('/feedback', methods=['POST'])
def feedback():
    summary = request.form['summary']
    quiz = request.form['quiz']
    fb = request.form['feedback']
    session = Session()
    session.add(Feedback(summary=summary, quiz=quiz, feedback=fb))
    session.commit()
    session.close()
    return "Thank you for your feedback!"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

