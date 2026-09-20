import esbuild from "esbuild";
import { spawn, spawnSync } from "node:child_process";
import { rm } from "node:fs/promises";
import { createRequire } from "node:module";

const watch = process.argv.includes("--watch");
const shell = process.platform === "win32";

const builds = [
  { format: "esm", outfile: "dist/index.js" },
  { format: "cjs", outfile: "dist/index.cjs" },
];

const options = {
  entryPoints: ["src/index.ts"],
  bundle: true,
  platform: "neutral",
  target: "es2020",
  sourcemap: true,
  minify: !watch,
  external: ["@huggingface/transformers"],
  logLevel: "info",
};

const tsc = createRequire(import.meta.url).resolve("typescript/bin/tsc");
const tscArgs = [tsc, "--project", "tsconfig.json", "--emitDeclarationOnly"];

const emitTypes = () => {
  const result = spawnSync("node", tscArgs, { stdio: "inherit", shell });
  if (result.error) {
    throw result.error;
  }
  if (result.status !== 0) {
    process.exit(result.status || 1);
  }
};

const watchTypes = () => {
  const child = spawn(
    "node",
    [...tscArgs, "--watch", "--preserveWatchOutput"],
    { stdio: "inherit", shell },
  );
  child.on("exit", (code) => {
    if (code !== 0 && code !== null) {
      process.exit(code);
    }
  });
  return child;
};

await rm("dist", { recursive: true, force: true });

if (watch) {
  const contexts = await Promise.all(
    builds.map((build) => esbuild.context({ ...options, ...build })),
  );
  await Promise.all(contexts.map((context) => context.watch()));
  const types = watchTypes();

  const stop = () => {
    types.kill("SIGTERM");
    Promise.all(contexts.map((context) => context.dispose())).finally(() =>
      process.exit(0),
    );
  };
  process.on("SIGINT", stop);
  process.on("SIGTERM", stop);

  console.log("Watching library bundles...");
} else {
  await Promise.all(
    builds.map((build) => esbuild.build({ ...options, ...build })),
  );
  emitTypes();
}
