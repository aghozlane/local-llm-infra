import { HfError, HfRateLimitError } from '../api/hf';
import { IconWarning } from './Icon';

interface ErrorPanelProps {
  readonly error: HfError;
  readonly retrying: boolean;
  readonly onRetry: () => void;
}

export function ErrorPanel({ error, retrying, onRetry }: ErrorPanelProps) {
  const isRateLimit = error instanceof HfRateLimitError;

  return (
    <div className="error-panel" role="alert">
      <IconWarning width={18} height={18} />
      <div className="error-body">
        <p className="error-message">{error.message}</p>
        {isRateLimit && retrying && (
          <p className="error-retry">Nouvelle tentative dans quelques secondes…</p>
        )}
        <button
          type="button"
          className="button button-secondary button-sm"
          onClick={onRetry}
          disabled={retrying}
        >
          Réessayer
        </button>
      </div>
    </div>
  );
}
