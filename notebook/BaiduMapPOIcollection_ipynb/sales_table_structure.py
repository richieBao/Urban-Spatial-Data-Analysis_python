# coding: utf-8
from sqlalchemy import BigInteger, Column, DateTime, MetaData, Table

metadata = MetaData()


t_sales = Table(
    'sales', metadata,
    Column('index', BigInteger, index=True),
    Column('idx', BigInteger),
    Column('date', DateTime),
    Column('exporting_country_ID', BigInteger)
)
