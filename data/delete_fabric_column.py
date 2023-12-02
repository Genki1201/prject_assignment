import pandas as pd

fabric_del_column_list = []
#消すつもりのデータが欲しいときは下のファイルないから消してどこかに+する
with open("D:\project_assignment\deep_fashion_explain\delete_fabric_column.txt", 'r') as file:
    for line in file:
        fabric_del_column_list.append(line.strip())

fabric_label = pd.read_csv("D:\project_assignment\deep_fashion_label\label_fabric.csv")
fabric_label.set_index(fabric_label.columns[0], inplace=True)

#使わない生地を削除する
fabric_del_column_label = fabric_label.copy()
for column in fabric_label.columns:
    if column in fabric_del_column_list:
        fabric_del_column_label.drop(column, axis=1, inplace=True)
        print("del: ", column)

fabric_del_column_label.to_csv("D:\project_assignment\deep_fashion_label\deleted_fabric.csv")

#似ている生地をまとめる
reducted_label = fabric_del_column_label.copy()

reducted_label[0]= (
    reducted_label['lace                         2']
    + reducted_label['crochet lace                 2']
    + reducted_label['crocheted lace               2']
    + reducted_label['cutout lace                  2']
    + reducted_label['embroidered lace             2']
    + reducted_label['eyelash lace                 2']
    + reducted_label['floral lace                  2']
    #+ reducted_label['knit lace                    2']
    + reducted_label['chiffon lace                 2']
    + reducted_label['lace layered                 2']
    + reducted_label['lace mesh                    2']
    + reducted_label['lace overlay                 2']
    + reducted_label['lace panel                   2']
    + reducted_label['lace paneled                 2']
    + reducted_label['lace pleated                 2']
    + reducted_label['lace print                   2']
    + reducted_label['lace sheer                   2']
    + reducted_label['lace-paneled                 2']
    + reducted_label['semi-sheer                   2']
    + reducted_label['sheer                        2']
    + reducted_label['sheer-paneled                2']
    + reducted_label['beaded sheer                 2']
    #+ reducted_label['lacy                         2']
    + reducted_label['organza                      2']
    #+ reducted_label['georgette                    2']
)
print('laceを作成しました')

reducted_label[1] = (
    reducted_label['knit                         2']
    +reducted_label['loose-knit                   2']
    +reducted_label['ribbed-knit                  2']
    +reducted_label['rib-knit                     2']
    +reducted_label['slub-knit                    2']
    +reducted_label['stretch-knit                 2']
    +reducted_label['cable knit                   2']
    +reducted_label['cable-knit                   2']
    +reducted_label['chunky knit                  2']
    +reducted_label['classic knit                 2']
    +reducted_label['crochet knit                 2']
    +reducted_label['eyelash knit                 2']
    +reducted_label['floral knit                  2']
    +reducted_label['fuzzy knit                   2']
    +reducted_label['heathered knit               2']
    +reducted_label['chunky                       2']
    +reducted_label['fair                         2']
    +reducted_label['fair isle                    2']
    +reducted_label['loop                         2']
    #+reducted_label['terry                        2']
    #+reducted_label['tweed                        2']
    + reducted_label['fuzzy                        2']
    + reducted_label['shaggy                       2']
)

print('knitを作成しました')

reducted_label[2] = (
    reducted_label['mesh                         2']
    + reducted_label['mesh overlay                 2']
    + reducted_label['mesh panel                   2']
    + reducted_label['mesh paneled                 2']
    + reducted_label['mesh-paneled                 2']
    + reducted_label['crochet mesh                 2']
    + reducted_label['embroidered mesh             2']
    + reducted_label['floral mesh                  2']
    + reducted_label['nets                         2']
    + reducted_label['netted                       2']
)

print('meshを作成しました')

reducted_label[3] =(
    reducted_label['denim                        2']
    + reducted_label['classic denim                2']
    + reducted_label['cuffed denim                 2']
    + reducted_label['bleached denim               2']
    + reducted_label['denim drawstring             2']
    + reducted_label['denim shirt                  2']
    + reducted_label['denim utility                2']
)
print('denimを作成しました')

reducted_label[4] =(
    reducted_label['leather                      2']
    + reducted_label['faux leather                 2']
    + reducted_label['leather paneled              2']
    + reducted_label['leather quilted              2']
    + reducted_label['leather-paneled              2']
)

print('leatherを作成しました')

reducted_label[5] =(
    reducted_label['cotton                       2']
    + reducted_label['classic cotton               2']
    + reducted_label['cotton drawstring            2']
    + reducted_label['cotton-blend                 2']
    + reducted_label['cotton knit                  2']
    + reducted_label['canvas                       2']
    + reducted_label['chambray                     2']
    + reducted_label['chambray drawstring          2']
    + reducted_label['pima                         2']
    + reducted_label['chino                        2']
    + reducted_label['scuba                        2']
    #+ reducted_label['print scuba                  2']
)

print('cottonを作成しました')

reducted_label[6] =(
    reducted_label['crepe                        2']
    + reducted_label['crepe woven                  2']
    + reducted_label['chiffon                      2']
    + reducted_label['chiffon layered              2']
    + reducted_label['chiffon shirt                2']
    + reducted_label['beaded chiffon               2']
    + reducted_label['foulard                      2']
    
)

print('chiffonを作成しました')

reducted_label[7] =(
    reducted_label['corduroy                     2']
    + reducted_label['feather                      2']
)
print('corduroyを作成しました')

reducted_label[8] =(
     reducted_label['fur                          2']
    + reducted_label['faux fur                     2']
    #+ reducted_label['shearling                    2']
)

print('furを作成しました')

reducted_label[9] =(
    reducted_label['metallic                     2']
    + reducted_label['satin                        2']
    #+ reducted_label['print satin                  2']
    + reducted_label['sleek                        2']
    #+ reducted_label['slick                        2']
    +reducted_label['nylon                        2']
)
print('metallicを作成しました')

reducted_label[10] =(
    reducted_label['stretch                      2']
    + reducted_label['oil                          2']
)
print('sportyを作成しました')

reducted_label[11] =(
    reducted_label['suede                        2']
    + reducted_label['faux suede                   2']
)

print('suedeを作成しました')

#作成した列だけを抜き出す
reducted_label= reducted_label[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]


#データ部分の0を0にそれ以外を1に変換
reducted_label = (~(reducted_label == 0)).astype(int)

reducted_label = reducted_label.astype(float)

row_sums = reducted_label.sum(axis=1) #生地の列において行方向の和を計算

#0以外のところは正規化する
for i in range(len(row_sums)):
    if row_sums.iloc[i] != 0:
        reducted_label.iloc[i] = reducted_label.iloc[i] / row_sums.iloc[i]
        
reducted_label.to_csv("D:\\project_assignment\\deep_fashion_label\\reducted_fabric.csv", index=True)
