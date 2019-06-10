import pymssql
import traceback
# 跟踪异常模块
import sys


class Dao:
    def __init__(self):
        pass

    # 未写入
    def get_connect(self):
        try:
            connect = pymssql.connect("127.0.0.1", port='1433', user='sa', password='123456', database="power_db",
                                     )
        except RuntimeError:
            f = open("log.txt", 'a')
            traceback.print_exc(file=f)
            f.flush()
            f.close()
            print("Connect failed")
        return connect

    # 插入数据
    def insertTable(self, sql):
        connct = self.get_connect()
        cursor = connct.cursor()

        try:
            cursor.execute(sql)
            connct.commit()

        except Exception:
            print("exception happened", Exception)
            f = open("log.txt", 'a')
            traceback.print_exc(file=f)
            f.flush()
            f.close()
            # 如果发生异常吗，则回滚
            connct.rollback()
        finally:
            connct.close()

    # 查询数据库
    def fetchone(self, sql):
        connect = self.get_connect()
        curor = connect.cursor()

        try:
            curor.execute(sql)
            result = curor.fetchone()
        except:
            traceback.print_exc()
            connect.rollback()
            f = open("log.txt", 'a')
            traceback.print_exc(file=f)
            f.flush()
            f.close()
        finally:
            connect.close()
        return result

    def fetchall(self, sql):
        connect = self.get_connect()
        cursor = connect()

        try:
            cursor.execute(sql)
            results = cursor.fetchall()
        except:
            # 输出异常信息
            info = sys.exc_info()
            print(info[0], ":", info[1])
            f = open("log.txt", 'a')
            traceback.print_exc(file=f)
            f.flush()
            f.close()
            connect.rollback()
        finally:
            connect.close()
        return results

    def delete(self, sql):
        connect = self.get_connect()
        cusor = connect.cursor()
        try:
            cusor.execute()
            connect.commit()

        except:
            info = sys.exc_info()
            print(info[0], ":", info[1])
            f = open("log.txt", 'a')
            traceback.print_exc(file=f)
            f.flush()
            f.close()
            connect.rollback()
        finally:
            connect.close()

    def update(self, sql):
        connect = self.get_connect()
        curor = connect.cursor()

        try:
            curor.execute(sql)
            connect.commit()
        except:
            connect.rollback()
        finally:
            connect.close()

dao = Dao()
sql = "INSERT INTO dbo.power VALUES ('1', '2', '3')"
dao.insertTable(sql)