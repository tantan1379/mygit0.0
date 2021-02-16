#include <iostream>

using namespace std;

class Animal {
	virtual void speak() = 0;
};

class Cat:public Animal {
public:
	virtual void speak() {
		cout << "小猫在说话" << endl;
	}
};

void test01() {
	Animal* animal = new Cat;
}


int main() {
	


	system("pause");
	return 0;
}