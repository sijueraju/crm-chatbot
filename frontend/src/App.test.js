import { render, screen } from '@testing-library/react';
import App from './App';

test('renders chat input', () => {
  render(<App />);
  const inputElement = screen.getByPlaceholderText(/ask about an order/i);
  expect(inputElement).toBeInTheDocument();
});
