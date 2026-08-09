"use strict";

const form = document.querySelector("#game-form");
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
  return `${model.run_id} · ${model.checkpoint}@${model.update} · source ${commit}`;
}

async function responseJSON(response) {
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.error || `Request failed with status ${response.status}`);
  }
  return payload;
}

async function loadSolutions() {
  try {
    const payload = await responseJSON(await fetch("/api/solutions", {headers: {Accept: "application/json"}}));
    solutionSelect.replaceChildren();
    for (const solution of payload.solutions) {
      const option = document.createElement("option");
      option.value = solution;
      option.textContent = solution;
      solutionSelect.append(option);
    }
    runtimeStatus.textContent = "CUDA inference ready";
    modelIdentity.textContent = modelLabel(payload.model);
    statusDot.classList.remove("loading", "error");
    solutionSelect.disabled = false;
    playButton.disabled = false;
    setMessage("Choose a solution and run a complete game.");
  } catch (error) {
    runtimeStatus.textContent = "Inference unavailable";
    statusDot.classList.remove("loading");
    statusDot.classList.add("error");
    setMessage(error.message, true);
  }
}

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
  playButton.disabled = true;
  solutionSelect.disabled = true;
  gameResult.hidden = true;
  setMessage(`Running CUDA inference for ${solutionSelect.value}…`);
  try {
    const response = await fetch("/api/games", {
      method: "POST",
      headers: {"Content-Type": "application/json", Accept: "application/json"},
      body: JSON.stringify({solution: solutionSelect.value}),
    });
    const game = await responseJSON(response);
    modelIdentity.textContent = modelLabel(game.model);
    await animateGame(game);
  } catch (error) {
    setMessage(error.message, true);
  } finally {
    playButton.disabled = false;
    solutionSelect.disabled = false;
  }
});

loadSolutions();
