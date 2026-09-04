import raw from '../data/hardware.json?raw';
import type { HardwareCatalog } from '../data/hardware.schema';
import { validateHardwareCatalog } from '../data/hardware.schema';

const parsed: unknown = JSON.parse(raw);
const validationErrors = validateHardwareCatalog(parsed as HardwareCatalog);

if (validationErrors.length > 0) {
  throw new Error('Catalogue matériel invalide : ' + validationErrors.join('; '));
}

export const catalog: HardwareCatalog = parsed as HardwareCatalog;
