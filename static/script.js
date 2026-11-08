(() => {
    const textarea = document.getElementById("user-input");
    const button = document.getElementById("detect-btn");
    const buttonLabel = button.querySelector(".button-label");
    const counter = document.getElementById("char-count");
    const resultBox = document.getElementById("result");
    const resultLabel = document.getElementById("result-label");
    const resultValue = document.getElementById("result-value");

    const MAX_CHARS = Number(textarea.getAttribute("maxlength")) || 500;

    // Names shown to the user for each class the model can return
    const EMOTION_NAMES = {
        anger: "Anger",
        fear: "Fear",
        joy: "Joy",
        love: "Love",
        sad: "Sadness",
        surprise: "Surprise",
    };

    let busy = false;

    function updateControls() {
        counter.textContent = `${textarea.value.length} / ${MAX_CHARS}`;
        button.disabled = busy || textarea.value.trim() === "";
    }

    function showLoading() {
        resultBox.hidden = false;
        resultBox.className = "result is-loading";
        resultBox.removeAttribute("data-emotion");
        resultLabel.textContent = "Analyzing";
        resultValue.textContent = "Reading your sentence...";
    }

    function showMood(mood) {
        const key = String(mood).toLowerCase();

        resultBox.hidden = false;
        resultBox.className = "result";
        resultBox.setAttribute("data-emotion", key);
        resultLabel.textContent = "Detected mood";
        resultValue.textContent = EMOTION_NAMES[key] || mood;
    }

    function showError(message) {
        resultBox.hidden = false;
        resultBox.className = "result is-error";
        resultBox.removeAttribute("data-emotion");
        resultLabel.textContent = "Something went wrong";
        resultValue.textContent = message;
    }

    async function detectMood() {
        const sentence = textarea.value.trim();
        if (sentence === "" || busy) return;

        busy = true;
        buttonLabel.textContent = "Detecting...";
        updateControls();
        showLoading();

        // stop waiting if the server does not answer
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 30000);

        try {
            const response = await fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ sentence }),
                signal: controller.signal,
            });

            const data = await response.json().catch(() => ({}));

            if (!response.ok) {
                throw new Error(data.error || `Server error (${response.status}).`);
            }

            showMood(data.mood);
        } catch (err) {
            console.error(err);
            const message =
                err.name === "AbortError"
                    ? "The server took too long to answer. Please try again."
                    : err.message || "Error detecting mood.";
            showError(message);
        } finally {
            clearTimeout(timeout);
            busy = false;
            buttonLabel.textContent = "Detect My Mood";
            updateControls();
        }
    }

    textarea.addEventListener("input", updateControls);
    button.addEventListener("click", detectMood);

    textarea.addEventListener("keydown", (event) => {
        if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
            event.preventDefault();
            detectMood();
        }
    });

    updateControls();
})();