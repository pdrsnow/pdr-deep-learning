import pandas as pd
from psycopg2 import Error as Error
from psycopg2.extensions import cursor as Cursor, connection as Connection
from psycopg2.pool import ThreadedConnectionPool


class PgMapper:
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
        except (Exception, Error) as error:
            print('Error while connecting to PostgreSQL', error)
            raise error

    def update(self, sql: str = '', params: dict = None):
        # 连接到数据库
        print(f'sql: {sql}')
        print(f'params: {params}')
        cursor: Cursor
        conn: Connection = self.dbpool.getconn()
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
        cursor: Cursor
        # 连接到数据库
        conn: Connection = self.dbpool.getconn()
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
        cursor: Cursor
        # 连接到数据库
        conn: Connection = self.dbpool.getconn()
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

    def qry_data_list(self, sql: str = '', params: dict = None):
        columns, records = self.select(sql, params)
        return [dict(zip(columns, record)) for record in records]

    def qry_data_frame(self, sql: str = '', params: dict = None):
        columns, records = self.select(sql, params)
        # 将结果转换为 DataFrame
        return pd.DataFrame(data=records, columns=columns)

# df = mapper.select('SELECT * FROM cs_traffic_monitor LIMIT 10', {})
# print(df.to_json(orient="records"))
# df.to_dict(orient='records')
