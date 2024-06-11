# from rest.mapper.mymapp import MyMapper
# from rest.mapper.pgmapp import PgMapper

"""
https://dev.mysql.com/doc/connector-python/en/connector-python-api-mysqlcursor-constructor.html
https://mysql.net.cn/doc/connectors/en/connector-python-api-mysqlcursor-column-names.html
"""

import pandas as pd
from mysql.connector import Error as _MyError
from mysql.connector.cursor import MySQLCursor as _MyCursor
from mysql.connector.pooling import MySQLConnectionPool, PooledMySQLConnection as _MyConnection


class Mapper:
    def upate(self, str='', params: dict = None) -> bool:
        pass

    def select(self, sql: str = '', params: dict = None) -> tuple[list, list]:
        pass

    def qry_data_one(self, sql: str = '', params: dict = None) -> dict:
        pass

    def qry_data_list(self, sql: str = '', params: dict = None) -> list:
        columns, records = self.select(sql, params)
        return [dict(zip(columns, record)) for record in records]

    def qry_data_frame(self, sql: str = '', params: dict = None) -> pd.DataFrame:
        columns, records = self.select(sql, params)
        # 将结果转换为 DataFrame
        return pd.DataFrame(data=records, columns=columns)


class MyMapper(Mapper):
    dbpool: MySQLConnectionPool = None

    def __init__(self, maxconn=5, host='127.0.0.1', port=3306, user='root', password='123456', database='mysql'):
        # 请替换为你的数据库名称
        dbconfig = {'host': host, 'port': port, 'user': user, 'password': password, 'database': database,
                    'autocommit': True}
        self.dbpool = MySQLConnectionPool(maxconn, **dbconfig)
        self._check_onnection()

    def _check_onnection(self):
        try:
            print(self.qry_data_one('SELECT VERSION()'))
        except (Exception, _MyError) as error:
            print('Error while connecting to MySQL', error)
            raise error

    def update(self, sql: str = '', params: dict = None):
        print(f'sql: {sql}')
        print(f'params: {params}')
        conn: _MyConnection
        cursor: _MyCursor
        # 连接到数据库
        with self.dbpool.get_connection() as conn:
            # 创建一个游标对象
            with conn.cursor() as cursor:
                # 执行SQL语句
                cursor.execute(sql, params)
                # 提交结果
                conn.commit()
                return True

    def select(self, sql: str = '', params: dict = None):
        """
        : (sql='SELECT * FROM tables WHERE id=%s AND name=%s', params=(1, 'tab_name'))

        : (sql='SELECT * FROM tables WHERE id=%(id)s AND name=%(name)s', params={'name':1, 'name':'tab_name'})
        :returns columns, records
        """
        print(f'sql: {sql}')
        print(f'params: {params}')
        conn: _MyConnection
        cursor: _MyCursor
        # 连接到数据库
        with self.dbpool.get_connection() as conn:
            # 创建一个游标对象
            with conn.cursor() as cursor:
                # 执行SQL语句
                cursor.execute(sql, params)
                # 获取字段
                columns = cursor.column_names
                # 获取查询结果
                records = cursor.fetchall()
                return columns, records

    def qry_data_one(self, sql: str = '', params: dict = None):
        """
        :returns {column: value}
        """
        print(f'sql: {sql}')
        print(f'params: {params}')
        conn: _MyConnection
        cursor: _MyCursor
        # 连接到数据库
        with self.dbpool.get_connection() as conn:
            # 创建一个游标对象
            with conn.cursor() as cursor:
                # 执行SQL语句
                cursor.execute(sql, params)
                # 获取字段
                columns = cursor.column_names
                # 获取查询结果
                record = cursor.fetchone()
                return dict(zip(columns, record)) if record else None


from psycopg2 import Error as _PgError
from psycopg2.extensions import cursor as _PgCursor, connection as _PgConnection
from psycopg2.pool import ThreadedConnectionPool


class PgMapper(Mapper):
    dbpool: ThreadedConnectionPool = None

    def __init__(self, maxconn=5, host='127.0.0.1', port=5432, user='postgres', password='postgres',
                 database='postgres'):
        # 请替换为你的数据库名称
        dbconfig = {'host': host, 'port': port, 'user': user, 'password': password, 'database': database}
        self.dbpool = ThreadedConnectionPool(minconn=1, maxconn=maxconn, **dbconfig)
        self._check_onnection()

    def _check_onnection(self):
        try:
            print(self.qry_data_one('SELECT VERSION()'))
        except (Exception, _PgError) as error:
            print('Error while connecting to PostgreSQL', error)
            raise error

    def update(self, sql: str = '', params: dict = None):
        # 连接到数据库
        print(f'sql: {sql}')
        print(f'params: {params}')
        cursor: _PgCursor
        conn: _PgConnection = self.dbpool.getconn()
        try:
            # 创建一个游标对象
            with conn.cursor() as cursor:
                # 执行SQL语句
                cursor.execute(sql, params)
                # 提交结果
                conn.commit()
                return True
        finally:
            # 归还到连接池
            self.dbpool.putconn(conn)

    def select(self, sql: str = '', params: dict = None):
        """
        : (sql='SELECT * FROM tables WHERE id=%s AND name=%s', params=(1, 'tab_name'))

        : (sql='SELECT * FROM tables WHERE id=%(id)s AND name=%(name)s', params={'name':1, 'name':'tab_name'})
        :returns columns, records
        """
        print(f'sql: {sql}')
        print(f'params: {params}')
        cursor: _PgCursor
        # 连接到数据库
        conn: _PgConnection = self.dbpool.getconn()
        try:
            # 创建一个游标对象
            with conn.cursor() as cursor:
                # 执行SQL语句
                cursor.execute(sql, params)
                # 获取字段
                columns = [x.name for x in cursor.description]
                # 获取查询结果
                records = cursor.fetchall()
                return columns, records
                # cursor.mogrify(sql, params)
                # print(f'totals: {cursor.rowcount}')
        finally:
            # 归还到连接池
            self.dbpool.putconn(conn)

    def qry_data_one(self, sql: str = '', params: dict = None):
        """
        :returns {column: value}
        """
        print(f'sql: {sql}')
        print(f'params: {params}')
        cursor: _PgCursor
        # 连接到数据库
        conn: _PgConnection = self.dbpool.getconn()
        try:
            # 创建一个游标对象
            with conn.cursor() as cursor:
                # 执行SQL语句
                cursor.execute(sql, params)
                # 获取字段
                columns = [x.name for x in cursor.description]
                # 获取查询结果
                record = cursor.fetchone()
                return dict(zip(columns, record)) if record else None
        finally:
            # 归还到连接池
            self.dbpool.putconn(conn)

# df = mapper.select('SELECT * FROM cs_traffic_monitor LIMIT 10', {})
# print(df.to_json(orient="records"))
# df.to_dict(orient='records')
