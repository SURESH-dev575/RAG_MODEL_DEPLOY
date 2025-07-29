async function sendQuestion() {
    const input = document.getElementById("user-input");
    const question = input.value;
    if (!question) return;

    const chatLog = document.getElementById("chat-log");
    chatLog.innerHTML += `<div><strong>You:</strong> ${question}</div>`;

    const response = await fetch("/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
    });

    const data = await response.json();
    chatLog.innerHTML += `<div><strong>Bot:</strong> ${data.answer}</div>`;
    input.value = "";
}
