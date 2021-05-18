#pragma once
#include <iostream>
#include <string>

using namespace std;

class Understudent {
private:
	string name;
	string ID;
	string password;
	float grade;
	string sex;
public:
	Understudent();
	Understudent(string n, string i, string pass, float g, string s);
	string get_name() {
		return name;
	}
	string get_id() {
		return ID;
	}
	string get_password() {
		return password;
	}
	float get_grade() {
		return grade;
	}
	string get_sex() {
		return sex;
	}
	void display();
	void set_password(string pass) {
		password = pass;
	}

	bool operator == (const Understudent& u)const {
		return ID == u.ID;
	}

	bool operator < (const Understudent& u)const {
		if (grade != u.grade) return grade < u.grade;
		else if (name.compare(u.name) != 0) return name < u.name;
		else if (ID.compare(u.ID) != 0) return ID < u.ID;
	}
};