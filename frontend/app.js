const form = document.getElementById("profile-form");
const submitBtn = document.getElementById("find-events-btn");
const statusEl = document.getElementById("status");

const orgsContainer = document.getElementById("orgs-container");
const orgsList = document.getElementById("orgs-list");

const eventsContainer = document.getElementById("events-container");
const eventsList = document.getElementById("events-list");

function setStatus(message, isError = false) {
  statusEl.textContent = message;
  statusEl.hidden = !message;
  statusEl.classList.toggle("error", isError);
}

function renderOrgs(profile) {
  orgsList.innerHTML = "";

  if (!profile.orgs || profile.orgs.length === 0) {
    const li = document.createElement("li");
    li.textContent = "No organizations found this time — try adding more detail.";
    orgsList.appendChild(li);
  } else {
    for (const org of profile.orgs) {
      const li = document.createElement("li");
      li.innerHTML = `<strong>${escapeHtml(org.name)}</strong>
        <div class="event-meta">${escapeHtml(org.category)}</div>`;
      orgsList.appendChild(li);
    }
  }

  orgsContainer.hidden = false;
}

function renderEvents(rankedEvents) {
  eventsList.innerHTML = "";

  for (const event of rankedEvents.events) {
    const li = document.createElement("li");
    li.innerHTML = `
      <a href="${escapeAttr(event.link)}" target="_blank" rel="noopener">${escapeHtml(event.title)}</a>
      <div class="event-meta">${formatDate(event.date)} &middot; ${escapeHtml(event.location)} &middot; ${escapeHtml(String(event.price))}</div>
    `;
    eventsList.appendChild(li);
  }

  eventsContainer.hidden = false;
}

function formatDate(isoString) {
  const parsed = new Date(isoString);
  if (Number.isNaN(parsed.getTime())) return isoString;
  return parsed.toLocaleString(undefined, {
    weekday: "short",
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

function escapeHtml(value) {
  const div = document.createElement("div");
  div.textContent = value ?? "";
  return div.innerHTML;
}

function escapeAttr(value) {
  return (value ?? "").replace(/"/g, "&quot;");
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const rawText = document.getElementById("raw-text").value.trim();
  const selectedCategories = Array.from(
    form.querySelectorAll('input[name="category"]:checked')
  ).map((el) => el.value);

  orgsContainer.hidden = true;
  eventsContainer.hidden = true;
  submitBtn.disabled = true;

  try {
    setStatus("Agent 1 is analyzing your interests and searching NYC orgs...");

    const profileResponse = await fetch("/agents/preference-profiler", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        raw_text: rawText,
        selected_categories: selectedCategories,
      }),
    });

    if (!profileResponse.ok) {
      throw new Error(`Preference profiler failed (${profileResponse.status})`);
    }

    const profile = await profileResponse.json();
    renderOrgs(profile);

    setStatus("Looking up events...");

    const eventsResponse = await fetch("/agents/event-retriever", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(profile),
    });

    if (!eventsResponse.ok) {
      throw new Error(`Event retriever failed (${eventsResponse.status})`);
    }

    const rankedEvents = await eventsResponse.json();
    renderEvents(rankedEvents);

    setStatus("");
  } catch (err) {
    console.error(err);
    setStatus(`Something went wrong: ${err.message}`, true);
  } finally {
    submitBtn.disabled = false;
  }
});
