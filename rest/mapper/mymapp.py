"""
https://dev.mysql.com/doc/connector-python/en/connector-python-api-mysqlcursor-constructor.html
https://mysql.net.cn/doc/connectors/en/connector-python-api-mysqlcursor-column-names.html
"""
import pandas as pd
from mysql.connector import Error
from mysql.connector.cursor import MySQLCursor as Cursor
from mysql.connector.pooling import MySQLConnectionPool, PooledMySQLConnection as Connection


class MyMapper:
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
        except (Exception, Error) as error:
            print('Error while connecting to MySQL', error)
            raise error

    def update(self, sql: str = '', params: dict = None):
        print(f'sql: {sql}')
        print(f'params: {params}')
        conn: Connection
        cursor: Cursor
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
        conn: Connection
        cursor: Cursor
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
        conn: Connection
        cursor: Cursor
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

    def qry_data_list(self, sql: str = '', params: dict = None):
        columns, records = self.select(sql, params)
        return [dict(zip(columns, record)) for record in records]

    def qry_data_frame(self, sql: str = '', params: dict = None):
        columns, records = self.select(sql, params)
        # 将结果转换为 DataFrame
        return pd.DataFrame(data=records, columns=columns)

# conn = MySQLConnection()
# cursor = conn.cursor()
# cursor.execute('sql', {})
