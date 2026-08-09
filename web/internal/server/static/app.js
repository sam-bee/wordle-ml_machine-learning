"use strict";

const form = document.querySelector("#game-form");
const modelSelect = document.querySelector("#model");
const solutionSelect = document.querySelector("#solution");
const playButton = document.querySelector("#play-button");
const message = document.querySelector("#message");
const gameResult = document.querySelector("#game-result");
const gameTitle = document.querySelector("#game-title");
const gameSummary = document.querySelector("#game-summary");
const board = document.querySelector("#board");
const runtimeStatus = document.querySelector("#runtime-status");
const statusDot = document.querySelector("#status-dot");
const modelIdentity = document.querySelector("#model-identity");
let activeRunID = "";
let controlsReady = false;

function setMessage(text, isError = false) {
  message.textContent = text;
  message.classList.toggle("error", isError);
}

function delay(milliseconds) {
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return Promise.resolve();
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

function modelLabel(model) {
  const commit = model.training_commit ? model.training_commit.slice(0, 8) : "unknown";
  const stage = model.stage ? `${model.stage} · ` : "";
  return `${model.run_id} · ${stage}${model.checkpoint}@${model.update} · source ${commit}`;
}

function modelOptionLabel(model) {
  return `${model.run_id} · ${model.stage} · ${model.checkpoint}@${model.update}`;
}

function setControlsDisabled(disabled) {
  modelSelect.disabled = disabled || !controlsReady;
  solutionSelect.disabled = disabled || !controlsReady;
  playButton.disabled = disabled || !controlsReady;
}

async function responseJSON(response) {
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.error || `Request failed with status ${response.status}`);
  }
  return payload;
}

async function loadPage() {
  try {
    const [modelsPayload, solutionsPayload] = await Promise.all([
      fetch("/api/models", {headers: {Accept: "application/json"}}).then(responseJSON),
      fetch("/api/solutions", {headers: {Accept: "application/json"}}).then(responseJSON),
    ]);
    if (!modelsPayload.models.length) throw new Error("No completed model runs are available");

    modelSelect.replaceChildren();
    for (const model of modelsPayload.models) {
      const option = document.createElement("option");
      option.value = model.run_id;
      option.textContent = modelOptionLabel(model);
      modelSelect.append(option);
    }
    activeRunID = modelsPayload.active.run_id;
    modelSelect.value = activeRunID;

    solutionSelect.replaceChildren();
    for (const solution of solutionsPayload.solutions) {
      const option = document.createElement("option");
      option.value = solution;
      option.textContent = solution;
      solutionSelect.append(option);
    }
    runtimeStatus.textContent = "CUDA inference ready";
    modelIdentity.textContent = modelLabel(modelsPayload.active);
    statusDot.classList.remove("loading", "error");
    controlsReady = true;
    setControlsDisabled(false);
    setMessage("Choose a model run and solution, then run a complete game.");
  } catch (error) {
    runtimeStatus.textContent = "Inference unavailable";
    statusDot.classList.remove("loading");
    statusDot.classList.add("error");
    setMessage(error.message, true);
  }
}

modelSelect.addEventListener("change", async () => {
  const requestedRunID = modelSelect.value;
  const previousRunID = activeRunID;
  setControlsDisabled(true);
  gameResult.hidden = true;
  runtimeStatus.textContent = "Loading model onto CUDA…";
  statusDot.classList.remove("error");
  statusDot.classList.add("loading");
  setMessage(`Loading ${requestedRunID} and warming on-device inference…`);
  try {
    const response = await fetch("/api/models", {
      method: "PUT",
      headers: {"Content-Type": "application/json", Accept: "application/json"},
      body: JSON.stringify({run_id: requestedRunID}),
    });
    const payload = await responseJSON(response);
    activeRunID = payload.model.run_id;
    modelSelect.value = activeRunID;
    modelIdentity.textContent = modelLabel(payload.model);
    runtimeStatus.textContent = "CUDA inference ready";
    statusDot.classList.remove("loading", "error");
    setMessage(`${activeRunID} is loaded on-device. Choose a solution to run inference.`);
  } catch (error) {
    modelSelect.value = previousRunID;
    runtimeStatus.textContent = previousRunID ? `${previousRunID} remains active` : "Inference unavailable";
    statusDot.classList.remove("loading");
    statusDot.classList.add("error");
    setMessage(error.message, true);
  } finally {
    setControlsDisabled(false);
  }
});

function feedbackClass(symbol) {
  if (symbol === "G") return "correct";
  if (symbol === "Y") return "present";
  return "absent";
}

function renderTurn(turn) {
  const row = document.createElement("div");
  row.className = "turn-row";
  row.setAttribute("aria-label", `Turn ${turn.turn}: ${turn.guess}, feedback ${turn.feedback}`);
  [...turn.guess].forEach((letter, index) => {
    const tile = document.createElement("span");
    tile.className = `tile ${feedbackClass(turn.feedback[index])}`;
    tile.textContent = letter;
    tile.style.animationDelay = `${index * 55}ms`;
    row.append(tile);
  });
  const meta = document.createElement("div");
  meta.className = "turn-meta";
  const candidates = document.createElement("strong");
  candidates.textContent = `Candidates ${turn.shortlist_size_before} → ${turn.shortlist_size_after}`;
  meta.append(candidates);
  if (turn.raw_top_guess !== turn.guess) {
    meta.append(`Raw preference ${turn.raw_top_guess} was unavailable; selected ${turn.guess}.`);
  } else {
    meta.append(`Selected ${turn.guess}.`);
  }
  row.append(meta);
  board.append(row);
}

async function animateGame(game) {
  board.replaceChildren();
  gameResult.hidden = false;
  gameTitle.textContent = `Solution: ${game.solution}`;
  gameSummary.textContent = "Playing…";
  for (const turn of game.turns) {
    renderTurn(turn);
    await delay(650);
  }
  gameSummary.textContent = game.solved ? `Solved in ${game.guesses}` : "Lost after six turns";
  setMessage(game.solved ? `The policy solved ${game.solution}.` : `The policy did not solve ${game.solution}.`);
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  setControlsDisabled(true);
  gameResult.hidden = true;
  setMessage(`Running CUDA inference for ${solutionSelect.value}…`);
  try {
    const response = await fetch("/api/games", {
      method: "POST",
      headers: {"Content-Type": "application/json", Accept: "application/json"},
      body: JSON.stringify({solution: solutionSelect.value}),
    });
    const game = await responseJSON(response);
    activeRunID = game.model.run_id;
    if ([...modelSelect.options].some((option) => option.value === activeRunID)) {
      modelSelect.value = activeRunID;
    }
    modelIdentity.textContent = modelLabel(game.model);
    await animateGame(game);
  } catch (error) {
    setMessage(error.message, true);
  } finally {
    setControlsDisabled(false);
  }
});

loadPage();
