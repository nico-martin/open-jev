import { spawnSync } from "node:child_process";
import { readFile, writeFile } from "node:fs/promises";

const registry = "https://registry.npmjs.org";
const otp = process.argv[2];
const versionBump = process.argv[3] || "patch";
const allowedVersionBumps = new Set(["patch", "minor", "major"]);

if (!otp || !allowedVersionBumps.has(versionBump)) {
  console.error("Usage: pnpm run publish <otp> [patch|minor|major]");
  process.exit(1);
}

const shell = process.platform === "win32";

const run = (command, args, options = {}) => {
  const result = spawnSync(command, args, {
    stdio: "inherit",
    shell,
    ...options,
  });

  if (result.status !== 0) {
    process.exit(result.status || 1);
  }
};

const bumpVersion = (version, bump) => {
  const parts = version.split(".").map((part) => parseInt(part, 10));
  if (parts.length !== 3 || parts.some((part) => Number.isNaN(part))) {
    throw new Error(`Cannot bump invalid version: ${version}`);
  }

  const [major, minor, patch] = parts;
  if (bump === "major") return `${major + 1}.0.0`;
  if (bump === "minor") return `${major}.${minor + 1}.0`;
  return `${major}.${minor}.${patch + 1}`;
};

const bumpPackageVersion = async () => {
  const packageJsonPath = "package.json";
  const packageJson = JSON.parse(await readFile(packageJsonPath, "utf8"));
  const currentVersion = packageJson.version;
  const nextVersion = bumpVersion(currentVersion, versionBump);

  packageJson.version = nextVersion;
  await writeFile(packageJsonPath, `${JSON.stringify(packageJson, null, 2)}\n`);
  console.log(
    `Version bumped ${currentVersion} -> ${nextVersion} (${versionBump})`,
  );

  return packageJson;
};

run("npm", ["whoami", "--registry", registry]);
run("pnpm", ["typecheck"]);
const packageJson = await bumpPackageVersion();
run("pnpm", ["build"]);

console.log(`Publishing ${packageJson.name}@${packageJson.version}...`);

run("npm", ["publish", "--registry", registry, "--otp", otp]);
