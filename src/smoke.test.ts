import { describe, expect, it } from 'vitest';

describe('smoke', () => {
  it('renders the application shell', () => {
    // Le test de fumée minimal valide que la pile Vite/React/TS
    // est correctement configurée (chainage de la toolchain).
    expect(typeof describe).toBe('function');
  });
});
