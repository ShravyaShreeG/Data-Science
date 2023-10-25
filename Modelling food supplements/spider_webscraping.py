import scrapy

class OrganicoSpider(scrapy.Spider):
    name = 'organico'
    allowed_domains = ['https://www.organico.ie','www.organico.ie']
    # start_urls = ['https://www.hollandandbarrett.ie/shop/vitamins-supplements/supplements/']
    page_count= 1

    def start_requests(self):
        yield scrapy.Request(url= 'https://www.organico.ie/supplements/', callback= self.parse, headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'})

    def parse(self, response):
        print("parse")
        product_container = response.xpath("//ol[@class= 'products list items product-items']")
        products = product_container.xpath(".//li[@class = 'item product product-item']")

        for product in products:
            product_title = product.xpath(".//a[@class = 'product-item-link']/text()").get()
            link = product.xpath(".//a[@class = 'product-item-link']/@href").get()
            # print(product_title)
            # print(link)
            yield response.follow(url=link, callback=self.parse_productlist, meta={'Product_title': product_title}, headers= {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'})

        # pagination = response.xpath("//ul[contains(@class,'items pages-items')]")
        # next_page_url = pagination.xpath(".//li[contains(@class,'item pages-item-next')]/a/@href").get()
        # print(next_page_url)

        if self.page_count != 37:  #37
            url = f'https://www.organico.ie/supplements?p={self.page_count}'
            yield response.follow(url=url, callback=self.parse)
            self.page_count += 1

    def parse_productlist(self, response):

        product_title = response.request.meta['Product_title']
        container = response.xpath("(//div[@class='product attribute description'])[1]")
        ingredients_text = container.xpath(".//strong[contains(text(),'Ingredients')]/parent::h3/following-sibling::p[1]/text()").get()

        if ingredients_text is None:
            ingredients_text = container.xpath(".//strong[contains(text(),'Ingredients')]/parent::h4/following-sibling::p[1]/text()").get()

        if ingredients_text is None:
            ingredients_text = container.xpath(".//strong[contains(text(),'Ingredients')]/following-sibling::text()").get()

        if ingredients_text is None:
            ingredients_text = container.xpath(".//h3[contains(text(),'Ingredients')]/following-sibling::p[1]/text()").get()

        if ingredients_text is None:
            ingredients_text = container.xpath(".//strong[contains(text(),'Ingredients')]/parent::p/following-sibling::p[1]/text()").get()

        yield {
            'Product_title': product_title,
            'Ingredients': ingredients_text,
        }