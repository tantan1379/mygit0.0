#include "System.h"
int System::underst_count = 0;
int System::ad_count = 0;

//登录界面
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

//退出系统
void System::exit_system() {
	cout << "****************感谢使用!******************" << endl;
	exit(0);
}

//本科生功能界面
void System::understudent_functionshow()
{
	cout << "***************************" << endl;
	cout << "1)查看个人信息" << endl;
	cout << "2)修改密码" << endl;
	cout << "3)返回上一级菜单!" << endl;
	cout << "*****************************" << endl;
	cout << "请选择你要进行的操作:";
}

//管理员功能界面
void System::administer_functionshow() {
	cout << "***************************" << endl;
	cout << "1)查看所有学生信息!" << endl;
	cout << "2)按成绩排序查看学生信息!" << endl;
	cout << "3)按姓名查找学生信息!" << endl;
	cout << "4)按学号查找学生信息!" << endl;
	cout << "5)录入学生信息" << endl;
	cout << "6)按学号删除学生信息" << endl;
	cout << "7)返回上一级菜单!" << endl;
	cout << "*****************************" << endl;
	cout << "请选择你要进行的操作:";
}

//管理员账号设置
void System::set_ad_account() {
	string name;
	string ID;
	string password;
	string password2;
	cout << endl << "请输入姓名:";
	cin >> name;
	cout << endl << "请输入ID：";
	cin >> ID;
	cout << endl << "请输入密码:" << endl;
	cin >> password;
	cout << endl << "请再次输入密码:" << endl;
	cin >> password2;
	while (password2 != password) {
		cout << "两次密码不一致，请再次确认：" << endl;
		cin >> password2;
	}
	Administer adm(name, ID, password);
	ad.push_back(adm);
	cout << "账号创建成功，请按任意键继续" << endl;
	cin.get();
	ad_count++;
	save_ad();
}

//管理员登录界面
void System::enter_ad_account() {
	int n;
	string udst_name;//查询姓名
	string udst_id;//查询id
	string id;
	string passw;
	list<Administer>::iterator iter;
	cout << "请输入用户账号：";
	cin >> id;
	int flag = 1;
	for (iter = ad.begin(); iter != ad.end(); iter++) {
		if (id == iter->get_id()) {
			flag = 0;
			break;
		}
	}
	if (flag) {
		cout << endl << "账户不存在！" << endl;
		return;
	}
	cout << endl << "请输入密码";
	cin >> passw;
	while (passw != iter->get_password()) {
		cout << endl << "密码错误，请重新输入：";
		cin >> passw;
	}
	cin.get();
	while (1) {
		system("cls");
		administer_functionshow();
		cin >> n;
		while (n < 1 || n>7) {
			cout << "请输入正确的选项(1-7)：";
			cin >> n;
		}
		switch (n) {
		case 1:
			input_underst_info();
			break;
		case 2:
			look_all_underst();
			break;
		case 3:
			look_all_underst_by_grade();
			break;
		case 4:
			cout << "请输入要查询的学生的名字：";
			cin >> udst_name;
			look_underst_by_name(udst_name);
			break;
		case 5:
			cout << "请输入要查询的学生的ID：";
			cin >> udst_id;
			look_underst_by_id(udst_id);
			break;
		case 6:
			cout << "请输入要删除学生的ID：";
			cin >> udst_id;
			delete_underst_by_id(udst_id);
			break;
		case 7:
			return;
			break;
		default:
			break;
		}
	}
}

//本科生登录界面
void System::enter_underst_account() {
	int n;
	list<Understudent>::iterator iter;
	string id;
	string passw;
	cout << "请输入用户账号：";
	cin >> id;
	int flag = 1;
	for (iter = underst.begin(); iter != underst.end(); iter++) {
		if (id == iter->get_id()) {
			flag = 0;
			break;
		}
	}
	if (flag) {
		cout << endl << "账户不存在！" << endl;
		return;
	}
	cout << endl << "请输入密码：";
	cin >> passw;
	while (passw != iter->get_password()) {
		cout << endl << "密码错误，请重新输入：";
		cin >> passw;
	}
	cin.get();
	while (1) {
		system("cls");
		understudent_functionshow();
		cin >> n;
		while (n < 1 || n>3) {
			cout << endl << "请输入正确的选项(1-3)：";
			cin >> n;
		}
		switch (n)
		{
		case 1:
			iter->display();
			break;
		case 2:
			change_password(id);
			break;
		case 3:
			return;
			break;
		default:
			break;
		}
	}
}

//保存管理员数据
void System::save_ad() {
	ofstream outfile("administer.dat");
	list<Administer>::iterator iter;
	outfile << ad_count << endl;
	for (iter == ad.begin(); iter != ad.end(); iter++) {
		outfile << iter->get_name() << "\t" << iter->get_id() << "\t" << iter->get_password() << endl;
	}
	outfile.close();
}

//保存本科生数据
void System::save_undst() {
	ofstream outfile("understudent.dat");
	list<Understudent>::iterator iter;
	outfile << underst_count << endl;
	for (iter == underst.begin(); iter != underst.end(); iter++) {
		outfile << iter->get_name() << "\t" << iter->get_id() << "\t"
			<< iter->get_password() << "\t" << iter->get_grade() << "\t" << iter->get_sex() << endl;
	}
	outfile.close();
}

//读取管理员数据
void System::load_ad() {
	ifstream infile("administer.dat");
	if (!infile) {
		cout << "无管理员资料！" << endl;
		return;
	}
	string name;
	string ID;
	string password;
	infile >> ad_count;
	infile.get();
	if (!ad.empty()) {
		ad.clear();
	}
	while (!infile.eof() || infile.peek() != EOF) {
		infile >> name >> ID >> password;
		Administer adm(name, ID, password);
		ad.push_back(adm);
		infile.get();
	}
	infile.close();
	cout << "成功读取管理员资料！" << endl;
}

//读取本科生数据
void System::load_undst() {
	ifstream infile("understudent.dat");
	if (!infile) {
		cout << "无本科生资料！" << endl;
		return;
	}
	string name;
	string ID;
	string password;
	float grade;
	string sex;
	infile >> underst_count;
	infile.get();//跳过回车字符
	while (!infile.eof() || infile.peek() != EOF) {
		infile >> name >> ID >> password >> grade >> sex;
		Understudent undst(name, ID, password, grade, sex);
		underst.push_back(undst);
		infile.get();//每读取一人就跳过回车字符继续读
	}
	infile.close();
	cout << "成功读取本科生资料！" << endl;
}

//管理员权限
//(2)查看所有本科生信息
void System::look_all_underst() {
	system("cls");
	if (underst.empty()) {
		cout << "无本科生数据！" << endl;
		system("pause");
		return;
	}
	list<Understudent>::iterator iter;
	cout << "姓名" << "\t" << "ID" << "\t" << "\t" << "性别" << "\t" << "绩点" << endl;
	for (iter = underst.begin(); iter != underst.end(); iter++) {
		cout<<iter->get_name()<<"\t"<<iter->get_id()<<"\t"<<iter->get_sex() << "\t" << iter->get_grade() << endl;
	}
	cout << endl << "学生总数为：" << underst_count << endl;
	system("pause");
}

//(3)按成绩排序显示本科生信息
void System::look_all_underst_by_grade() {
	system("cls");
	if (underst.empty()) {
		cout << "无本科生数据！" << endl;
		system("pause");
		return;
	}
	list<Understudent> underst_copy;
	for (list<Understudent>::iterator iter = underst.begin(); iter != underst.end(); iter++) {
		underst_copy.push_back(*iter);
	}
	underst_copy.sort();


	cout << "姓名" << "\t" << "ID" << "\t" << "\t" << "性别" << "\t" << "绩点" << endl;
	for (list<Understudent>::iterator iter = underst.begin(); iter != underst.end(); iter++) {
		cout<<iter->get_name()<<"\t"<<iter->get_id()<<"\t"<<iter->get_sex() << "\t" << iter->get_grade() << endl;
	}
	cout << endl << "学生总数为：" << underst_count << endl;
	system("pause");
}

