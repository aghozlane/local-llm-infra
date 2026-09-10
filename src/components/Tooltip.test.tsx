import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { createRoot, type Root } from 'react-dom/client';
import { act } from 'react';
import { Tooltip } from './Tooltip';

const DELAY_MS = 2000;

declare global {
  // eslint-disable-next-line no-var
  var IS_REACT_ACT_ENVIRONMENT: boolean | undefined;
}

globalThis.IS_REACT_ACT_ENVIRONMENT = true;

let cleanupQueue: Array<() => void> = [];

function renderTooltip(label = 'Explication') {
  const container = document.createElement('div');
  document.body.appendChild(container);
  let root: Root | null = null;
  act(() => {
    root = createRoot(container);
    root.render(<Tooltip label={label}>Cible</Tooltip>);
  });
  cleanupQueue.push(() => {
    act(() => {
      root?.unmount();
    });
    container.remove();
  });
  return container;
}

function pointerEnter(element: HTMLElement) {
  element.dispatchEvent(new MouseEvent('mouseenter', { bubbles: true }));
  element.dispatchEvent(new MouseEvent('mouseover', { bubbles: true, relatedTarget: document.body }));
}

function pointerLeave(element: HTMLElement) {
  element.dispatchEvent(new MouseEvent('mouseleave', { bubbles: true }));
  element.dispatchEvent(new MouseEvent('mouseout', { bubbles: true, relatedTarget: document.body }));
}

function getTrigger(container: HTMLElement) {
  const trigger = container.querySelector('.tooltip-trigger');
  if (!(trigger instanceof HTMLElement)) {
    throw new Error('Tooltip trigger not found');
  }
  return trigger;
}

function getTooltip(container: HTMLElement) {
  const tooltip = container.querySelector('.tooltip-box');
  if (!(tooltip instanceof HTMLElement)) {
    throw new Error('Tooltip box not found');
  }
  return tooltip;
}

describe('Tooltip', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    cleanupQueue = [];
  });

  afterEach(() => {
    cleanupQueue.forEach((cleanup) => cleanup());
    vi.useRealTimers();
    document.body.innerHTML = '';
  });

  it('does not show immediately on mouseenter', () => {
    const container = renderTooltip();
    const trigger = getTrigger(container);

    act(() => {
      pointerEnter(trigger);
    });

    expect(container.querySelector('.tooltip-box')).toBeNull();
  });

  it('shows after 2 seconds on mouseenter', () => {
    const container = renderTooltip();
    const trigger = getTrigger(container);

    act(() => {
      pointerEnter(trigger);
      vi.advanceTimersByTime(DELAY_MS);
    });

    const tooltip = getTooltip(container);
    expect(tooltip.classList.contains('is-visible')).toBe(true);
    expect(trigger.getAttribute('aria-describedby')).toBe(tooltip.id);
  });

  it('hides on mouseleave', () => {
    const container = renderTooltip();
    const trigger = getTrigger(container);

    act(() => {
      pointerEnter(trigger);
      vi.advanceTimersByTime(DELAY_MS);
      pointerLeave(trigger);
    });

    expect(container.querySelector('.tooltip-box')).toBeNull();
    expect(trigger.hasAttribute('aria-describedby')).toBe(false);
  });

  it('hides on Escape key', () => {
    const container = renderTooltip();
    const trigger = getTrigger(container);

    act(() => {
      pointerEnter(trigger);
      vi.advanceTimersByTime(DELAY_MS);
      trigger.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
    });

    expect(container.querySelector('.tooltip-box')).toBeNull();
  });

  it('clears pending timer on mouseleave before delay', () => {
    const container = renderTooltip();
    const trigger = getTrigger(container);

    act(() => {
      pointerEnter(trigger);
      vi.advanceTimersByTime(1000);
      pointerLeave(trigger);
      vi.advanceTimersByTime(DELAY_MS);
    });

    expect(container.querySelector('.tooltip-box')).toBeNull();
  });
});
