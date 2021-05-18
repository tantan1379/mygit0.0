#include "Understudent.h"
#include <iostream>

using namespace std;

Understudent::Understudent() {
	name = "default";
	ID = "default";
	password = "123";
	sex = "default";
	grade = 0;
}

Understudent::Understudent(string n, string i, string pass, float g, string s) :name(n), ID(i), password(pass), grade(g), sex(s) {}

void Understudent::display() {
	cout << "******************" << endl;
	cout << "* 姓名:" << name << endl;
	cout << "* 学号:" << ID << endl;
	cout << "* 性别:" << sex << endl;
	cout << "* 绩点:" << grade << endl;
	cout << "******************" << endl;
}