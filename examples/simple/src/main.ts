import "./style.css";
import { OpenJev, choice, noul, score } from "../../../src/index";
import type { Answer } from "../../../src/index";

const infoButton = getEl<HTMLButtonElement>("info");
const initButton = getEl<HTMLButtonElement>("init");
const decideButton = getEl<HTMLButtonElement>("decide");
const statusEl = getEl<HTMLParagraphElement>("status");
const modelEl = getEl<HTMLParagraphElement>("model");
const progressEl = getEl<HTMLProgressElement>("progress");
const stateEl = getEl<HTMLTextAreaElement>("state");
const answersEl = getEl<HTMLDivElement>("answers");
const logEl = getEl<HTMLPreElement>("log");

const questions = {
  area: choice("Which product area is the message about?", [
    "fees & charges",
    "pin & security",
    "refund & dispute",
    "top-up",
    "exchange & fiat",
    "atm & cash",
    "transfer",
    "card",
    "account & identity",
    "other",
  ]),
  sentiment: score("How positive is the sentiment of this message?", [
    "very negative",
    "negative",
    "neutral",
    "positive",
    "very positive",
  ]),
  refund: noul("The customer is asking for a refund."),
};

const config = { dtype: "q4f16" } as const;
let jev: OpenJev | null = null;

infoButton.addEventListener("click", async () => {
  const info = await OpenJev.info(config);
  const mb = (info.downloadSize / 1024 / 1024).toFixed(0);
  modelEl.textContent = `Model: ${info.device}/${info.dtype}, cached=${info.isCached}, download=${mb} MB`;
  writeLog(`info: ${info.files.join(", ")}`);
});

initButton.addEventListener("click", async () => {
  if (jev) {
    writeLog("already loaded");
    return;
  }

  setStatus("loading");
  progressEl.value = 0;
  initButton.disabled = true;

  try {
    jev = await OpenJev.load({
      ...config,
      onProgress: ({ progress, loaded, total }) => {
        progressEl.value = progress;
        const mb = (n: number) => (n / 1024 / 1024).toFixed(0);
        setStatus(`downloading ${mb(loaded)} / ${mb(total)} MB`);
      },
    });
    progressEl.value = 1;
    setStatus(`ready (${jev.runtime.device}/${jev.runtime.dtype})`);
    writeLog("load: ready");
    decideButton.disabled = false;
  } catch (error) {
    setStatus("error");
    writeLog(`load failed: ${(error as Error).message}`);
    initButton.disabled = false;
  }
});

decideButton.addEventListener("click", async () => {
  if (!jev) {
    return;
  }

  setStatus("deciding");
  const started = performance.now();
  try {
    const answers = await jev.decide(stateEl.value, questions);
    const took = Math.round(performance.now() - started);
    setStatus(`ready (${jev.runtime.device}/${jev.runtime.dtype})`);
    writeLog(
      `decide (${took} ms, ${jev.countTokens(stateEl.value)} state tokens): area=${answers.area.choice}, sentiment=${answers.sentiment.level}, refund=${answers.refund.answer}`,
    );
    renderAnswers(answers);
  } catch (error) {
    setStatus("error");
    writeLog(`decide failed: ${(error as Error).message}`);
  }
});

function renderAnswers(answers: Record<string, Answer>): void {
  answersEl.innerHTML = "";
  for (const [key, answer] of Object.entries(answers)) {
    const block = document.createElement("div");
    block.className = "answer";
    const title = document.createElement("h3");
    block.appendChild(title);

    if (answer.type === "noul") {
      title.textContent = `${key}: ${answer.answer ? "yes" : "no"} (p(yes) = ${answer.probability.toFixed(3)})`;
      block.appendChild(bar("yes", answer.probability, answer.answer));
      block.appendChild(bar("no", 1 - answer.probability, !answer.answer));
    } else {
      const headline =
        answer.type === "choice"
          ? `${key}: ${answer.choice} (confidence ${answer.confidence.toFixed(3)})`
          : `${key}: ${answer.level} (score ${answer.score.toFixed(2)}, confidence ${answer.confidence.toFixed(3)})`;
      title.textContent = headline;
      const best = answer.type === "choice" ? answer.choice : answer.level;
      for (const [option, p] of Object.entries(answer.probabilities)) {
        block.appendChild(bar(option, p, option === best));
      }
    }

    answersEl.appendChild(block);
  }
}

function bar(label: string, value: number, best: boolean): HTMLDivElement {
  const row = document.createElement("div");
  row.className = best ? "bar best" : "bar";
  row.innerHTML = `<span>${label}</span><span><i style="width:${(value * 100).toFixed(1)}%"></i></span><span>${(value * 100).toFixed(1)}%</span>`;
  return row;
}

function setStatus(value: string): void {
  statusEl.textContent = `Status: ${value}`;
}

function writeLog(line: string): void {
  const stamp = new Date().toLocaleTimeString();
  logEl.textContent = `[${stamp}] ${line}\n${logEl.textContent}`;
}

function getEl<T extends HTMLElement>(id: string): T {
  const element = document.getElementById(id);
  if (!element) {
    throw new Error(`Missing #${id}`);
  }
  return element as T;
}
