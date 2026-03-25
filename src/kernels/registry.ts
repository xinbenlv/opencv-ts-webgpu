import type { KernelDescriptor } from '../backends/interface.ts';

/**
 * Global registry for compute kernels.
 * Kernel descriptors are registered at module load time (side-effect imports).
 */
class KernelRegistryImpl {
  private readonly _kernels = new Map<string, KernelDescriptor>();

  register(descriptor: KernelDescriptor): void {
    if (this._kernels.has(descriptor.name)) {
      throw new Error(`Kernel "${descriptor.name}" is already registered.`);
    }
    this._kernels.set(descriptor.name, descriptor);
  }

  get(name: string): KernelDescriptor {
    const k = this._kernels.get(name);
    if (!k) {
      throw new Error(
        `Kernel "${name}" not found. Registered: [${[...this._kernels.keys()].join(', ')}]`,
      );
    }
    return k;
  }

  has(name: string): boolean {
    return this._kernels.has(name);
  }

  list(): readonly KernelDescriptor[] {
    return [...this._kernels.values()];
  }

  get size(): number {
    return this._kernels.size;
  }
}

export const KernelRegistry = new KernelRegistryImpl();
