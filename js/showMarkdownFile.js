function renderMarkdownFile(filePath, targetElementId) {
    const targetElement = document.getElementById(targetElementId);
    if (!targetElement) {
        console.error("Target element not found.");
        return;
    }

    const xhr = new XMLHttpRequest();
    xhr.open("GET", filePath, true);

    xhr.onreadystatechange = function () {
        if (xhr.readyState === 4 && xhr.status === 200) {
            const markdownContent = xhr.responseText;
            targetElement.innerHTML = marked.parse(markdownContent);
        }
    };

    xhr.send();
}