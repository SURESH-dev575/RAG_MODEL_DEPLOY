const chatLog = document.getElementById("chat-log");
const userInput = document.getElementById("user-input");
const sendBtn = document.getElementById("send-btn");

function appendMessage(who, text, id = null) {
    const div = document.createElement("div");
    div.className = who === "You" ? "msg you" : "msg bot";
    if (id) div.id = id;

    const avatar = who === "You" ? "👤" : "🤖";
    
    div.innerHTML = `
        <div class="avatar">${avatar}</div>
        <div class="message-content">${text}</div>
    `;
    
    chatLog.appendChild(div);
    chatLog.scrollTop = chatLog.scrollHeight;
    return div;
}

async function sendQuestion() {
    const question = userInput.value.trim();
    if (!question) return;

    // Add user message
    appendMessage("You", question);
    userInput.value = "";
    
    // Disable input while waiting
    userInput.disabled = true;
    sendBtn.disabled = true;

    // Add loading message
    const loadingId = "loading-" + Date.now();
    appendMessage("Bot", '<span class="typing-indicator">Thinking (this may take a minute on the first run)...</span>', loadingId);

    try {
        const resp = await fetch("/ask", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question }),
        });

        // Remove loading message
        document.getElementById(loadingId).remove();

        // FIX FOR THE JSON ERROR: Read as text first to check if Render sent an HTML timeout page
        const textResponse = await resp.text();
        
        try {
            // Try to parse it as JSON
            const data = JSON.parse(textResponse);
            appendMessage("Bot", data.answer);
        } catch (jsonError) {
            // If it fails to parse, Render likely timed out or crashed
            console.error("Server HTML Response:", textResponse);
            appendMessage("Bot", "⚠️ **Server Error:** The AI took too long to load or ran out of memory on the free tier. Try asking again in a few seconds once it wakes up.");
        }

    } catch (err) {
        document.getElementById(loadingId)?.remove();
        appendMessage("Bot", "⚠️ **Connection Error:** " + err.message);
    } finally {
        userInput.disabled = false;
        sendBtn.disabled = false;
        userInput.focus();
    }
}

sendBtn.addEventListener("click", sendQuestion);
userInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") sendQuestion();
});

// Upload functionality
async function uploadFile() {
    const fi = document.getElementById("file-input");
    const msg = document.getElementById("upload-msg");
    
    if (!fi.files || fi.files.length === 0) {
        msg.className = "status-msg status-error";
        msg.textContent = "Please select a file first.";
        return;
    }

    const file = fi.files[0];
    const form = new FormData();
    form.append("file", file);

    msg.className = "status-msg";
    msg.textContent = "Uploading and learning...";

    try {
        const resp = await fetch("/upload", {
            method: "POST",
            body: form
        });
        
        const textResponse = await resp.text();
        try {
            const data = JSON.parse(textResponse);
            if (data.ok) {
                msg.className = "status-msg";
                msg.textContent = "✅ " + data.msg;
                fi.value = ""; // clear input
            } else {
                msg.className = "status-msg status-error";
                msg.textContent = "❌ " + (data.msg || "Unknown error");
            }
        } catch (e) {
            msg.className = "status-msg status-error";
            msg.textContent = "❌ Server crashed or timed out during upload.";
        }
    } catch (err) {
        msg.className = "status-msg status-error";
        msg.textContent = "❌ Upload error: " + err.message;
    }
}

document.getElementById("upload-btn").addEventListener("click", uploadFile);
