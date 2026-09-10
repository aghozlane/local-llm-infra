import { useEffect, useId, useRef, useState } from 'react';
import type { ReactNode, KeyboardEvent } from 'react';

interface TooltipProps {
  readonly children: ReactNode;
  readonly label: string;
  readonly position?: 'top' | 'bottom';
}

const DELAY_MS = 2000;

export function Tooltip({ children, label, position = 'top' }: TooltipProps) {
  const [visible, setVisible] = useState(false);
  const timeoutRef = useRef<number | null>(null);
  const tooltipId = useId();

  const show = () => {
    if (timeoutRef.current !== null) {
      clearTimeout(timeoutRef.current);
    }
    timeoutRef.current = setTimeout(() => {
      setVisible(true);
    }, DELAY_MS);
  };

  const hide = () => {
    if (timeoutRef.current !== null) {
      clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
    }
    setVisible(false);
  };

  useEffect(() => {
    return () => {
      if (timeoutRef.current !== null) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, []);

  const handleKeyDown = (event: KeyboardEvent<HTMLSpanElement>) => {
    if (event.key === 'Escape') {
      event.preventDefault();
      hide();
    }
  };

  const positionClass = position === 'bottom' ? 'tooltip-box--bottom' : '';

  return (
    <span
      className="tooltip-trigger"
      tabIndex={0}
      onMouseEnter={show}
      onMouseLeave={hide}
      onFocus={show}
      onBlur={hide}
      onKeyDown={handleKeyDown}
      aria-describedby={visible ? tooltipId : undefined}
    >
      {children}
      {visible && (
        <span
          id={tooltipId}
          role="tooltip"
          className={`tooltip-box is-visible ${positionClass}`}
        >
          {label}
        </span>
      )}
    </span>
  );
}
