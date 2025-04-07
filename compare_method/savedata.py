import xlrd
from xlutils.copy import copy


def savedata(file, row, column, data):
    oldWB = xlrd.open_workbook(file)
    newWB = copy(oldWB)
    newBs = newWB.get_sheet('Sheet1')
    newBs.write(row, column, data[0])
    newBs.write(30 + row, column, data[1])
    newBs.write(60 + row, column, data[2])
    newBs.write(90 + row, column, data[3])

    newWB.save(file)


def savedata_order(file, row, column, data):
    oldWB = xlrd.open_workbook(file)
    newWB = copy(oldWB)
    newBs = newWB.get_sheet('Sheet1')
    newBs.write(row, column, data[0])
    newBs.write(43 + row, column, data[1])
    newBs.write(86 + row, column, data[2])
    newBs.write(129 + row, column, data[3])

    newWB.save(file)
