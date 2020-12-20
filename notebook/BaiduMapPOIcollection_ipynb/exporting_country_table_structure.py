# coding: utf-8
from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()
metadata = Base.metadata


class ExportingCountry(Base):
    __tablename__ = 'exporting_country'

    index = Column(Integer, primary_key=True)
    exporting_country_ID = Column(Integer)
    exporting_country_name = Column(String(32))
