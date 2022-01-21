import pymssql
import os

class MSSQL:
    def __init__(self,host,user,pwd,db):
        self.host = host
        self.user = user
        self.pwd = pwd
        self.db = db

    def __GetConnect(self):
        """
        得到连接信息
        返回: conn.cursor()
        """
        if not self.db:
            raise(NameError,"没有设置数据库信息")
        self.conn = pymssql.connect(host=self.host,user=self.user,password=self.pwd,database=self.db,charset="utf8")
        cur = self.conn.cursor()
        if not cur:
            raise(NameError,"连接数据库失败")
        else:
            return cur

    def ExecQuery(self,sql):
        """
        执行查询语句
        返回的是一个包含tuple的list，list的元素是记录行，tuple的元素是每行记录的字段

        """
        cur = self.__GetConnect()
        cur.execute(sql)
        resList = cur.fetchall()

        #查询完毕后必须关闭连接
        self.conn.close()
        return resList

    def ExecNonQuery(self,sql):
        """
        执行非查询语句

        调用示例：
            cur = self.__GetConnect()
            cur.execute(sql)
            self.conn.commit()
            self.conn.close()
        """
        cur = self.__GetConnect()
        cur.execute(sql)
        self.conn.commit()
        self.conn.close()


def save_picture(ms, name, path):
    '''
    写入图片文件如数据库
    :param ms: 实例化后的数据库类
    :param name: 图片的名字
    :param path: 图片的路径
    :return:
    '''
    ms.ExecNonQuery(r"insert into face(name,path) values({}, {})".format(name, path))


def remove_blank(r):
    '''
    去掉元组的空格并将其转为列表
    :param r: 查询后的元组
    :return: 去掉空格后的列表
    '''
    tu = []
    for i in r:
        tu.append(str(i[0]))
    for t in range(len(r)):
        tu[t] = tu[t].replace(' ', '')
    return tu


def get_path(ms):
    '''
    得到数据库中的图片信息
    :param ms: 实例化后的数据库类
    :return: 数据库中的图片的路径信息
    '''
    r = remove_blank(ms.ExecQuery("select path from face"))
    print(r)
    return r


def remove_infor_blank(r):
    tu = []
    for i in r:
        tu.append(str(i))
    for t in range(len(r)):
        tu[t] = tu[t].replace(' ', '')
    return tu


def get_columns(ms):
    infor = ms.ExecQuery("select name from syscolumns where id=object_id('face')")
    return infor


def get_information(ms):
    '''
    获取数据库中face表的所有信息
    :param ms:
    :return:
    '''
    infor = ms.ExecQuery("select * from face")
    information = []
    for i in infor:
        information.append(remove_infor_blank(i))
    return information


def main():
    ms = MSSQL(host="127.0.0.1:1433",user="sa",pwd="123456",db="SuperMarket")
    dir = r'E:\人脸识别\数据集\data'
    # get_path(ms)
    s = ms.ExecQuery('select * from face')
    s1 = get_information(ms)
    # s = get_columns(ms)
    # print(s)
    # print(s1)
    # name = '司马懿'
    # i = 'E:\人脸识别\数据集\small\林允儿子\林允儿_0.jpg'
    # ms.ExecNonQuery(r"insert into face(name,path) values('{}', '{}')".format(name, i))
    # L = []
    # for root, dirs, files in os.walk(dir):
    #     for file in files:
    #         if os.path.splitext(file)[1] == '.jpg':
    #             L.append(os.path.join(root, file))
    # for i in L:
    #     a = i.split('\\')
    #     name = a[-2]
    #     ms.ExecNonQuery(r"insert into face(name,path) values('{}', '{}')".format(name, i))


# if __name__ == '__main__':
#     main()
