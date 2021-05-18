#include "System.h"

int System::underst_count = 0;
int System::ad_count = 0;

void System::load_interface() {//用于显示界面并连接其他功能函数
	int i;
	while (1) {
		system("cls");
		load_ad();//读取管理员数据
		load_undst();//读取本科生数据
		cout << "********************" << endl;
		cout << "1)开通管理员账户!" << endl;
		cout << "2)管理员身份登陆!" << endl;
		cout << "3)本科生身份登陆!" << endl;
		cout << "4)退出系统!" << endl;
		cout << "********************" << endl;
		cout << "请输入操作:";
		cin >> i;
		while (i < 1 || i>4) {
			cout << "请输入正确的序号!" << endl;
			cout << "请重新输入:";
			cin >> i;
		}
		switch (i) {
		case 1:
			set_ad_account();
			break;
		case 2:
			enter_ad_account();
			break;
		case 3:
			enter_underst_account();
			break;
		case 4:
			exit_system();
			break;
		default:
			break;
		}
	}
}

void System::exit_system(){
	cout << "****************感谢使用!******************" << endl;
	exit(0);
}

void System::understudent_functionshow()
{
	cout << "***************************" << endl;
	cout << "1)查看个人信息" << endl;
	cout << "2)修改密码" << endl;
	cout << "3)返回上一级菜单!" << endl;
	cout << "*****************************" << endl;
	cout << "请选择你要进行的操作:";
}

void System::administer_functionshow() {
	cout << "***************************" << endl;
	cout << "1)查看所有学生信息!" << endl;
	cout << "2)按姓名查找学生信息!" << endl;
	cout << "3)按学号查找学生信息!" << endl;
	cout << "4)录入学生信息" << endl;
	cout << "5)按学号删除学生信息" << endl;
	cout << "6)返回上一级菜单!" << endl;
	cout << "7)按成绩排序!" << endl;
	cout << "*****************************" << endl;
	cout << "请选择你要进行的操作:";
}

void System::set_ad_account() {
	string name;
	string ID;
	string password;
	string password2;
	cout << endl << "请输入姓名:";
	cin >> name;
	cout << endl << "请输入ID：";
	cin >> ID;
	cout << endl << "请输入密码";
	cin >> password;
	cout << endl << "请再次输入密码";
	cin >> password2;
	while(password2 != password) {
		cout << "两次密码不一致，请再次确认：" << endl;
		cin >> password2;
	}
	Administer adm(name, ID, password);
	ad.push_back(adm);
}