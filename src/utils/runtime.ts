import type { OpenJevDevice, OpenJevDtype, OpenJevRuntime } from "../types";

type GpuAdapterLike = { features: { has(name: string): boolean } };
type NavigatorWithGpu = Navigator & {
  gpu?: { requestAdapter(): Promise<GpuAdapterLike | null> };
};

let fp16Support: Promise<boolean> | null = null;

function isNode(): boolean {
  const proc = (globalThis as { process?: { versions?: { node?: string } } })
    .process;
  return typeof proc?.versions?.node === "string";
}

export function isWebGpuAvailable(): boolean {
  return (
    typeof navigator !== "undefined" &&
    typeof (navigator as NavigatorWithGpu).gpu !== "undefined"
  );
}

export function isWebGpuFp16Supported(): Promise<boolean> {
  if (!fp16Support) {
    fp16Support = (async () => {
      if (!isWebGpuAvailable()) {
        return false;
      }
      try {
        const adapter = await (
          navigator as NavigatorWithGpu
        ).gpu!.requestAdapter();
        return adapter?.features.has("shader-f16") ?? false;
      } catch {
        return false;
      }
    })();
  }

  return fp16Support;
}

/**
 * Resolve `"auto"` device/dtype to concrete values.
 *
 * - device: `webgpu` when the runtime exposes WebGPU, `cpu` in Node.js,
 *   otherwise `wasm`.
 * - dtype: the family's preferred WebGPU variant when `shader-f16` is
 *   available, otherwise its fallback (`q4` unless the family says otherwise).
 */
export async function resolveRuntime(
  options: {
    device?: OpenJevDevice | "auto";
    dtype?: OpenJevDtype | "auto";
  },
  family: {
    webgpuDtype: OpenJevDtype;
    fallbackDtype?: OpenJevDtype;
    dtypes?: readonly OpenJevDtype[];
  },
): Promise<Pick<OpenJevRuntime, "device" | "dtype">> {
  const requestedDevice = options.device ?? "auto";
  const requestedDtype = options.dtype ?? "auto";

  let device: OpenJevDevice;
  if (requestedDevice !== "auto") {
    device = requestedDevice;
  } else if (isWebGpuAvailable()) {
    device = "webgpu";
  } else if (isNode()) {
    device = "cpu";
  } else {
    device = "wasm";
  }

  let dtype: OpenJevDtype;
  if (requestedDtype !== "auto") {
    dtype = requestedDtype;
  } else if (device === "webgpu" && (await isWebGpuFp16Supported())) {
    dtype = family.webgpuDtype;
  } else {
    dtype = family.fallbackDtype ?? "q4";
  }

  if (family.dtypes && !family.dtypes.includes(dtype)) {
    throw new Error(
      `This model ships ${family.dtypes.join(", ")} weights only; dtype "${dtype}" is not available.`,
    );
  }

  return { device, dtype };
}
