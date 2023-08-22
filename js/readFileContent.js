// 定义一个函数来获取文件内容
function readFileContent(filename, callback) {
    const xhr = new XMLHttpRequest();

    xhr.open("GET", filename, true);

    xhr.onreadystatechange = function () {
        if (xhr.readyState === 4 && xhr.status === 200) {
            const fileContent = xhr.responseText;
            // 调用回调函数，并传递文件内容
            callback(fileContent);
        }
    };

    xhr.send();
}

