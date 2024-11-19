#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <map>
#include <vector>

int main() {
    // 定义变量
    std::string name;
    std::string type;
    std::vector<float> anchors;
    int num_anchors = 0;
    int dw = 0;

    std::ifstream file("parameters.txt");

    if (!file.is_open()) {
        std::cerr << "无法打开文件" << std::endl;
        return 1;
    }

    std::string line;
    std::string content;

    // 读取文件内容到一个字符串
    while (std::getline(file, line)) {
        content += line + "\n";
    }

    file.close();

    std::istringstream stream(content);
    std::string currentLine;
    std::map<std::string, std::string> keyValueMap;

    while (std::getline(stream, currentLine)) {
        // 去除空行
        if (currentLine.empty()) continue;

        // 处理“:”分隔的键值对
        size_t pos = currentLine.find(":");
        if (pos != std::string::npos) {
            std::string key = currentLine.substr(0, pos);
            std::string value = currentLine.substr(pos + 1);

            // 去掉多余的空格和引号
            key.erase(0, key.find_first_not_of(" \t"));
            key.erase(key.find_last_not_of(" \t") + 1);
            value.erase(0, value.find_first_not_of(" \t\""));
            value.erase(value.find_last_not_of(" \t\"") + 1);

            if (key == "anchors") {
                anchors.emplace_back(std::stof(value));
            } else {
                keyValueMap[key] = value;
            }

            std::cout << key << ": " << value << std::endl;
        }
    }

    // 根据键值对设置变量
    if (keyValueMap.find("name") != keyValueMap.end()) {
        name = keyValueMap["name"];
    }
    if (keyValueMap.find("type") != keyValueMap.end()) {
        type = keyValueMap["type"];
    }
    if (keyValueMap.find("num_anchors") != keyValueMap.end()) {
        std::istringstream(keyValueMap["num_anchors"]) >> num_anchors;
    }
    if (keyValueMap.find("dw") != keyValueMap.end()) {
        std::istringstream(keyValueMap["dw"]) >> dw;
    }

    // 输出变量值
    std::cout << "Name: " << name << std::endl;
    std::cout << "Type: " << type << std::endl;
    std::cout << "Num Anchors: " << num_anchors << std::endl;
    std::cout << "Anchors:";
    for (float anchor : anchors) {
        std::cout << " " << anchor;
    }
    std::cout << std::endl;

    return 0;
}