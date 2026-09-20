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
 * - dtype: `fp16` on WebGPU with `shader-f16`, otherwise `q4`.
 */
export async function resolveRuntime(options: {
  device?: OpenJevDevice | "auto";
  dtype?: OpenJevDtype | "auto";
}): Promise<OpenJevRuntime> {
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
    dtype = "fp16";
  } else {
    dtype = "q4";
  }

  return { device, dtype };
}
