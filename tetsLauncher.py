import ReadAndJoin as Raj

#read data from navdata.csv and commands.csv to panda dataframes and join them by time
#test prints
print(Raj.cmds)
print(Raj.cmds.index)
print(Raj.cmds.columns)

print(Raj.navData)
print(Raj.navData.index)
print(Raj.navData.columns)
print(len(Raj.navDataColumnNames))

print(Raj.merged)
print(Raj.merged.index)
print(Raj.merged.columns)

print(Raj.merged.index)

isCmdTimeDuplicated = Raj.cmds["time"].duplicated()
isCmdTimeDuplicated.to_csv("duplicatedCmd.tsv", sep='\t')

isNavDataTimeDuplicated = Raj.navData["time"].duplicated()
isNavDataTimeDuplicated.to_csv("duplicatedNav.tsv", sep='\t')

isMergedTimeDuplicated = Raj.merged["time"].duplicated()
isMergedTimeDuplicated.to_csv("duplicatedMerged.tsv", sep='\t')
