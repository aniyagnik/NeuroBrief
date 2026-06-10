import neuroBriefLogo from '../logo.png';

export default function Header() {
  return (
    <>
      <img
        src={neuroBriefLogo}
        alt="NeuroBrief Logo"
        className="w-32 h-32 md:w-48 md:h-48 mb-4 object-contain"
      />
      <h1 className="text-3xl md:text-4xl font-bold text-blue-700 mb-2">🧠 NeuroBrief</h1>
      <p className="text-gray-700 text-sm md:text-base max-w-2xl mb-6 px-4">
        Upload a video or paste a YouTube link. Status updates show while each step runs.
      </p>
    </>
  );
}
