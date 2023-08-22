// 获取输入框和列表项元素
// 注意，你的输入框的ID应该是`searchInput`
const searchInput = document.getElementById("searchInput");
const listItems = document.querySelectorAll("#list li");

// 当输入框内容变化时执行的代码
searchInput.addEventListener("input", function () {
    // 获取输入框内容并转换为小写
    const searchTerm = searchInput.value.toLowerCase();

    // 遍历每个列表项
    listItems.forEach(item => {
        // 获取列表项的文本内容并转换为小写
        const text = item.textContent.toLowerCase();

        // 查找关键词在文本中的位置
        const index = text.indexOf(searchTerm);

        if (index !== -1) {
            // 创建一个新字符串，标注关键词
            const markedText = item.textContent.slice(0, index)
                + "<span class='highlight'>"
                + item.textContent.slice(index, index + searchTerm.length)
                + "</span>"
                + item.textContent.slice(index + searchTerm.length);

            // 将标注后的文本赋值给列表项
            item.innerHTML = markedText;

            // 显示匹配的列表项
            item.style.display = "block";
        } else {
            // 隐藏不匹配的列表项,当然也可以
            item.style.display = "none";
        }
    });
});

// 监听键盘按键事件
document.addEventListener("keydown", function (event) {
    // 按下的键是 "/"
    if (event.key === "/") {
        // 将焦点放在搜索框上
        searchInput.focus();

        // 添加扩展样式，放大搜索框
        searchInput.classList.add("expanded");

        // 阻止默认的 "/" 键行为（防止在输入框中输入斜杠）
        event.preventDefault();
    }
});

// 监听输入框失去焦点事件
searchInput.addEventListener("blur", function () {
    searchInput.classList.remove("expanded") /*取消扩大状态*/
});

// 当检测到ESC输入的时候,退出并且缩小搜索框
document.addEventListener("keydown", function (event) {
    // 按下的是`Escape`
    if (event.key === "Escape") {
        // 添加扩展样式，放大搜索框
        searchInput.classList.remove("expanded");
    }
})