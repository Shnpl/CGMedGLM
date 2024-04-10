from modules.chain_module import MainChain
def main():
    chain = MainChain()
    x = "晚期胃癌，从去年开始出现流口水，进食少，上腹部不适，今年4月份于当地县人民医院做B超检查发现有肝胆管结石，予口服药消炎药治疗，未见减轻，后7月份做胃镜检查发现幽门管堵塞，并发现疑是肿瘤病灶，3天后做手术治疗放置人工幽门管，并做病理切片，确诊为胃癌，扩散至大网膜，手术未切除肿瘤。7天后伤口愈合拆线，现在总感觉腹部隐痛，腰做B超检查有肾积水。"
    chain.invoke(x)
if __name__ == '__main__':
    main()