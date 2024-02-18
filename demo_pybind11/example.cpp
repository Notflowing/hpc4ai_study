#include <pybind11/pybind11.h>
#include <string>
#include <memory>

namespace py = pybind11;

int add(int i, int j) {
    return i + j;
}

struct Pet {
    Pet(const std::string& name): name(name) {}
    Pet(const int a) {name = std::to_string(a);}
    void setName(const std::string& name_) {name = name_;}
    const std::string &getName() const {return name;}
    virtual ~Pet() = default;

    void set(int age_) {age = age_;}
    void set(const std::string& name_) {name = name_;}

    std::string name;
    int age{0};
};

struct Dog: Pet {
    Dog(const std::string& name): Pet(name) {}
    std::string bark() const {return "woof!";}
};

PYBIND11_MODULE(example, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    // m.def("add", &add, "A function that adds two numbers");
    m.def("add", &add, "A function which adds two numbers", py::arg("i") = 1, py::arg("j") = 2);
    m.def("lambdaadd", [](int i, int j) {return i + j;}, "Lambda function which adds two numbers");

    m.attr("the_answer") = 42;
    py::object world = py::cast("world");
    m.attr("what") = world;

    py::class_<Pet>(m, "Pet")
        .def(py::init<const std::string&>())
        .def(py::init<const int>())
        .def("setName", &Pet::setName)
        .def("getName", &Pet::getName)
        .def("__repr__", [](const Pet& a) {return "<example.Pet named '" + a.name + "'>";})
        .def("set", py::overload_cast<int>(&Pet::set), "Set the Pet's age")
        .def("set", py::overload_cast<const std::string&>(&Pet::set), "Set the Pet's name");
    
    py::class_<Dog, Pet>(m, "Dog")
        .def(py::init<const std::string&>())
        .def("bark", &Dog::bark);
    
    m.def("pet_store", []() {return std::unique_ptr<Pet>(new Dog("Molly"));});
}

