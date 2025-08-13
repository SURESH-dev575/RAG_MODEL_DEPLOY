async function appendMessage(who, text) {
    const chatLog = document.getElementById("chat-log");
    const div = document.createElement("div");
    div.className = who === "You" ? "msg you" : "msg bot";
    div.innerHTML = `<strong>${who}:</strong> ${text}`;
    chatLog.appendChild(div);
    chatLog.scrollTop = chatLog.scrollHeight;
}

async function sendQuestion() {
    const input = document.getElementById("user-input");
    const question = input.value.trim();
    if (!question) return;
    await appendMessage("You", question);
    input.value = "";

    try {
        const resp = await fetch("/ask", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question }),
        });
        const data = await resp.json();
        await appendMessage("Bot", data.answer);
    } catch (err) {
        await appendMessage("Bot", "Error contacting server: " + err.message);
    }
}

document.getElementById("send-btn").addEventListener("click", sendQuestion);
document.getElementById("user-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter") sendQuestion();
});

// Upload
async function uploadFile() {
    const fi = document.getElementById("file-input");
    const msg = document.getElementById("upload-msg");
    if (!fi.files || fi.files.length === 0) {
        msg.textContent = "No file selected.";
        return;
    }
    const file = fi.files[0];
    const form = new FormData();
    form.append("file", file);

    msg.textContent = "Uploading...";
    try {
        const resp = await fetch("/upload", {
            method: "POST",
            body: form
        });
        const data = await resp.json();
        if (data.ok) {
            msg.textContent = data.msg;
        } else {
            msg.textContent = "Upload failed: " + (data.msg || "Unknown error");
        }
    } catch (err) {
        msg.textContent = "Upload error: " + err.message;
    }
}

document.getElementById("upload-btn").addEventListener("click", uploadFile);
