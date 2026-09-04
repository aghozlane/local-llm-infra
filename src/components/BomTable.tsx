import type { BomLine } from '../engine/recommendations';
import { formatCurrencyEur } from '../lib/format';

interface BomTableProps {
  readonly bom: readonly BomLine[];
}

export function BomTable({ bom }: BomTableProps) {
  const total = bom.reduce((sum, line) => sum + line.totalEur, 0);

  return (
    <table className="bom-table">
      <thead>
        <tr>
          <th scope="col">Libellé</th>
          <th scope="col" className="bom-qty">
            Qté
          </th>
          <th scope="col" className="bom-price">
            Prix unitaire
          </th>
          <th scope="col" className="bom-price">
            Total
          </th>
        </tr>
      </thead>
      <tbody>
        {bom.map((line) => (
          <tr key={line.label}>
            <td>{line.label}</td>
            <td className="bom-qty">{line.quantity.toLocaleString('fr-FR')}</td>
            <td className="bom-price">{formatCurrencyEur(line.unitPriceEur)}</td>
            <td className="bom-price">{formatCurrencyEur(line.totalEur)}</td>
          </tr>
        ))}
      </tbody>
      <tfoot>
        <tr>
          <th scope="row" colSpan={3}>
            Total configuration
          </th>
          <td className="bom-price bom-total">{formatCurrencyEur(total)}</td>
        </tr>
      </tfoot>
    </table>
  );
}
