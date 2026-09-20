import { spawn, spawnSync } from "node:child_process";

const shell = process.platform === "win32";

const initialBuild = spawnSync("node", ["scripts/build.mjs"], {
  stdio: "inherit",
  shell,
});

if (initialBuild.error) {
  throw initialBuild.error;
}

if (initialBuild.status !== 0) {
  process.exit(initialBuild.status || 1);
}

const commands = [
  ["node", ["scripts/build.mjs", "--watch"], {}],
  ["pnpm", ["dev"], { cwd: "examples/simple" }],
];

const processes = commands.map(([command, args, options]) =>
  spawn(command, args, { stdio: "inherit", shell, ...options }),
);

const stop = (exitCode = 0) => {
  for (const child of processes) {
    child.kill("SIGTERM");
  }
  process.exit(exitCode);
};

for (const child of processes) {
  child.on("exit", (code) => {
    if (code !== 0 && code !== null) stop(code);
  });
}

process.on("SIGINT", () => stop());
process.on("SIGTERM", () => stop());
