import { BYTES_PER_GIB } from '../engine/formulas';
import { bpwFor } from '../engine/quantization';
import type { QuantName } from '../engine/types';
import type { ResolvedModel } from '../api/hf';
import { formatDecimal } from '../lib/format';

export interface FormState {
  readonly quant: QuantName;
  readonly simultaneous: string;
  readonly totalUsers: string;
  readonly reqPerUserPerMin: string;
  readonly inputTokens: string;
  readonly outputTokens: string;
  readonly contextMax: string;
  readonly ttftTargetSec: string;
  readonly minTpsPerUser: string;
  readonly safetyMargin: string;
}

interface LoadFormProps {
  readonly form: FormState;
  readonly onChange: (patch: Partial<FormState>) => void;
  readonly model: ResolvedModel | null;
}

const QUANTS: readonly QuantName[] = ['FP16', 'Q8_0', 'Q6_K', 'Q5_K_M', 'Q4_K_M'];
export const CONTEXT_OPTIONS = [4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576];
const SAFETY_OPTIONS = [
  { value: '0.1', label: '10 %' },
  { value: '0.15', label: '15 %' },
  { value: '0.2', label: '20 %' },
  { value: '0.3', label: '30 %' },
  { value: '0.4', label: '40 %' },
];

function quantWeightLabel(quant: QuantName, totalParams: number): string {
  const bpw = bpwFor(quant);
  const gib = (totalParams * bpw) / 8 / BYTES_PER_GIB;
  return `≈ ${formatDecimal(gib)} Go`;
}

export function LoadForm({ form, onChange, model }: LoadFormProps) {
  const makeNumberHandler = (field: keyof FormState) =>
    (event: React.ChangeEvent<HTMLInputElement>) => {
      onChange({ [field]: event.target.value });
    };

  const enforceMin = (field: keyof FormState, min: number) =>
    (event: React.FocusEvent<HTMLInputElement>) => {
      const value = Number(event.target.value);
      if (!Number.isFinite(value) || value < min) {
        onChange({ [field]: String(min) });
      }
    };

  return (
    <form className="stack" onSubmit={(e) => e.preventDefault()}>
      <fieldset className="form-section stack">
        <legend className="section-title">Quantification</legend>
        <div className="quant-grid">
          {QUANTS.map((quant) => {
            const weight = model ? quantWeightLabel(quant, model.totalParams) : null;
            return (
              <label
                key={quant}
                className={`quant-option${form.quant === quant ? ' is-selected' : ''}`}
              >
                <input
                  type="radio"
                  name="quant"
                  value={quant}
                  checked={form.quant === quant}
                  onChange={() => onChange({ quant })}
                />
                <span className="quant-name">{quant}</span>
                {weight && <span className="quant-weight">{weight}</span>}
              </label>
            );
          })}
        </div>
      </fieldset>

      <fieldset className="form-section stack">
        <legend className="section-title">Charge</legend>
        <div className="form-grid">
          <div className="field">
            <label htmlFor="simultaneous" className="field-label">
              Séquences simultanées
            </label>
            <input
              id="simultaneous"
              type="number"
              min={1}
              className="input"
              value={form.simultaneous}
              onChange={makeNumberHandler('simultaneous')}
              onBlur={enforceMin('simultaneous', 1)}
            />
          </div>

          <div className="field">
            <label htmlFor="totalUsers" className="field-label">
              Utilisateurs totaux
            </label>
            <input
              id="totalUsers"
              type="number"
              min={1}
              className="input"
              value={form.totalUsers}
              onChange={makeNumberHandler('totalUsers')}
              onBlur={enforceMin('totalUsers', 1)}
            />
          </div>

          <div className="field">
            <label htmlFor="reqPerUserPerMin" className="field-label">
              Requêtes / utilisateur / min
            </label>
            <input
              id="reqPerUserPerMin"
              type="number"
              min={1}
              className="input"
              value={form.reqPerUserPerMin}
              onChange={makeNumberHandler('reqPerUserPerMin')}
              onBlur={enforceMin('reqPerUserPerMin', 1)}
            />
          </div>

          <div className="field">
            <label htmlFor="inputTokens" className="field-label">
              Tokens d’entrée
            </label>
            <input
              id="inputTokens"
              type="number"
              min={1}
              className="input"
              value={form.inputTokens}
              onChange={makeNumberHandler('inputTokens')}
              onBlur={enforceMin('inputTokens', 1)}
            />
          </div>

          <div className="field">
            <label htmlFor="outputTokens" className="field-label">
              Tokens de sortie
            </label>
            <input
              id="outputTokens"
              type="number"
              min={1}
              className="input"
              value={form.outputTokens}
              onChange={makeNumberHandler('outputTokens')}
              onBlur={enforceMin('outputTokens', 1)}
            />
          </div>

          <div className="field">
            <label htmlFor="contextMax" className="field-label">
              Contexte max
            </label>
            <div className="select-wrapper">
              <select
                id="contextMax"
                className="input select"
                value={form.contextMax}
                onChange={(e) => onChange({ contextMax: e.target.value })}
              >
                {CONTEXT_OPTIONS.map((value) => (
                  <option key={value} value={value}>
                    {value.toLocaleString('fr-FR')}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </div>
      </fieldset>

      <fieldset className="form-section stack">
        <legend className="section-title">Cibles &amp; marge</legend>
        <div className="form-grid">
          <div className="field">
            <label htmlFor="ttftTargetSec" className="field-label">
              TTFT max (s)
            </label>
            <input
              id="ttftTargetSec"
              type="number"
              min={0}
              step={0.1}
              className="input"
              value={form.ttftTargetSec}
              onChange={makeNumberHandler('ttftTargetSec')}
              placeholder="aucune cible"
            />
          </div>

          <div className="field">
            <label htmlFor="minTpsPerUser" className="field-label">
              Débit min / utilisateur (tok/s)
            </label>
            <input
              id="minTpsPerUser"
              type="number"
              min={0}
              step={0.1}
              className="input"
              value={form.minTpsPerUser}
              onChange={makeNumberHandler('minTpsPerUser')}
              placeholder="aucune cible"
            />
          </div>

          <div className="field">
            <label htmlFor="safetyMargin" className="field-label">
              Marge de sécurité
            </label>
            <div className="select-wrapper">
              <select
                id="safetyMargin"
                className="input select"
                value={form.safetyMargin}
                onChange={(e) => onChange({ safetyMargin: e.target.value })}
              >
                {SAFETY_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </div>
      </fieldset>
    </form>
  );
}
