import { useCallback, useEffect, useId, useRef, useState } from 'react';
import type { HfError, ModelSearchHit, ResolvedModel } from '../api/hf';
import { formatNumber } from '../lib/format';
import { IconSearch, IconWarning } from './Icon';
import { ErrorPanel } from './ErrorPanel';

interface ModelSearchProps {
  readonly query: string;
  readonly onQueryChange: (value: string) => void;
  readonly hits: readonly ModelSearchHit[];
  readonly isSearching: boolean;
  readonly selectedModel: ResolvedModel | null;
  readonly error: HfError | null;
  readonly retrying: boolean;
  readonly onSelectHit: (id: string) => void;
  readonly onResolveDirect: () => void;
  readonly onRetry: () => void;
}

export function ModelSearch({
  query,
  onQueryChange,
  hits,
  isSearching,
  selectedModel,
  error,
  retrying,
  onSelectHit,
  onResolveDirect,
  onRetry,
}: ModelSearchProps) {
  const id = useId();
  const listId = `${id}-listbox`;
  const inputId = `${id}-input`;
  const [isOpen, setIsOpen] = useState(false);
  const [activeIndex, setActiveIndex] = useState(-1);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLUListElement>(null);

  const trimmed = query.trim();
  const canResolveDirect = trimmed.length >= 3;
  const showHits = isOpen && hits.length > 0;

  useEffect(() => {
    setIsOpen(hits.length > 0 && document.activeElement === inputRef.current);
    setActiveIndex(-1);
  }, [hits]);

  const closeList = useCallback(() => {
    setIsOpen(false);
    setActiveIndex(-1);
  }, []);

  const handleKeyDown = (event: React.KeyboardEvent<HTMLInputElement>) => {
    if (!showHits) return;
    if (event.key === 'ArrowDown') {
      event.preventDefault();
      setActiveIndex((prev) => (prev + 1) % hits.length);
    } else if (event.key === 'ArrowUp') {
      event.preventDefault();
      setActiveIndex((prev) => (prev - 1 + hits.length) % hits.length);
    } else if (event.key === 'Enter') {
      event.preventDefault();
      const hit = hits[activeIndex];
      if (hit !== undefined) {
        onSelectHit(hit.id);
        closeList();
      }
    } else if (event.key === 'Escape') {
      closeList();
      inputRef.current?.focus();
    }
  };

  const handleSelect = (id: string) => {
    onSelectHit(id);
    closeList();
  };

  return (
    <section className="form-section stack" aria-labelledby={inputId}>
      <h2 className="section-title">Modèle</h2>

      <div className="field">
        <label htmlFor={inputId} className="field-label">
          Rechercher un modèle Hugging Face
        </label>
        <div className="input-wrapper">
          <IconSearch className="input-icon" />
          <input
            ref={inputRef}
            id={inputId}
            type="text"
            className="input"
            value={query}
            onChange={(e) => onQueryChange(e.target.value)}
            onFocus={() => setIsOpen(hits.length > 0)}
            onKeyDown={handleKeyDown}
            placeholder="ex. Qwen/Qwen3-30B-A3B"
            autoComplete="off"
            aria-autocomplete="list"
            aria-controls={listId}
            aria-expanded={showHits}
            aria-activedescendant={
              activeIndex >= 0 ? `${id}-option-${activeIndex}` : undefined
            }
            role="combobox"
          />
          {isSearching && <span className="input-spinner" aria-hidden="true" />}
        </div>
        <span className="field-hint">
          3 caractères minimum. Coller un identifiant <code>org/modèle</code> puis cliquer sur
          « Résoudre cet identifiant ».
        </span>
      </div>

      {canResolveDirect && !selectedModel && (
        <button
          type="button"
          className="button button-secondary button-sm"
          onClick={onResolveDirect}
        >
          Résoudre cet identifiant
        </button>
      )}

      {showHits && (
        <ul
          ref={listRef}
          id={listId}
          role="listbox"
          className="autocomplete-list"
          aria-label="Résultats de recherche"
        >
          {hits.map((hit, index) => {
            const isActive = index === activeIndex;
            return (
              <li
                key={hit.id}
                id={`${id}-option-${index}`}
                role="option"
                aria-selected={isActive}
                className={`autocomplete-option${isActive ? ' is-active' : ''}`}
                onMouseEnter={() => setActiveIndex(index)}
                onClick={() => handleSelect(hit.id)}
              >
                <span className="autocomplete-name">{hit.name}</span>
                <span className="autocomplete-id">{hit.id}</span>
              </li>
            );
          })}
        </ul>
      )}

      {selectedModel && (
        <div className="model-summary card card-subtle stack-sm">
          <div className="cluster">
            <span className="model-name">{selectedModel.name}</span>
            <span className="model-id text-secondary">{selectedModel.id}</span>
          </div>
          <div className="cluster">
            <span className="badge badge-neutral">
              {formatNumber(selectedModel.totalParams)} paramètres
            </span>
            {selectedModel.isMoe && (
              <span className="badge badge-info">MoE</span>
            )}
            {(selectedModel.kvHeadsInferred || selectedModel.headDimInferred) && (
              <span className="badge badge-warning">valeurs déduites</span>
            )}
            {selectedModel.moeTreatedAsDense && (
              <span className="badge badge-warning" title="Architecture d’expert non détectée">
                <IconWarning width={12} height={12} />
                MoE traité comme dense
              </span>
            )}
            {selectedModel.resolvedFromBaseModel && (
              <span
                className="badge badge-warning"
                title={`Architecture déduite du modèle de base : ${selectedModel.baseModelId ?? ''}`}
              >
                Architecture du modèle de base
              </span>
            )}
          </div>
        </div>
      )}

      {error && <ErrorPanel error={error} retrying={retrying} onRetry={onRetry} />}
    </section>
  );
}
