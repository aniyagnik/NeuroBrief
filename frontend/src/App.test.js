import { render, screen } from '@testing-library/react';
import App from './App';

test('renders NeuroBrief', () => {
  render(<App />);
  expect(screen.getByText(/NeuroBrief/i)).toBeInTheDocument();
});
