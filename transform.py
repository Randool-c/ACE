def transf(_input):
    finData = []
    for item in _input:
        for node in item['links']:
            tag = 0
            if finData == []:
                finData.append({'sourceIP': item['id'], 'destinationIP': node})
            else:
                for data in finData:
                    if node == data['sourceIP']:
                        tag = 1
                        break
                # if node in list(map(lambda d: d['sourceIP'], finData)):
                #     tag = 1
                if tag == 0:
                    finData.append({'sourceIP': item['id'], 'destinationIP': node})
    return finData
            

def main():
    _input = [{'id': '1', 'links': ['2', '3', '3']}, {'id': '2', 'links': ['1']}, {'id': '3', 'links': ['1', '1']}]
    Data = transf(_input)
    print(Data)


if __name__ == '__main__':
    main()
