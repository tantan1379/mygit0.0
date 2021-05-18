#include "Administer.h"
#include <iostream>

using namespace std;

Administer::Administer() { //无参构造方法实现
	name = "admin", ID = "0000", password = "123";
}

Administer::Administer(string n, string i, string pass) :name(n), ID(i), password(pass) {}//有参构造方法实现

void Administer::display() {
	cout << endl << "******************" << endl;
	cout << endl << "* 姓名：" << name;
	cout << endl << "* 账号：" << ID;
	cout << endl << "******************" << endl;
}

