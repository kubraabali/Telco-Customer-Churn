# Telco Customer Churn

![churn](https://i.pinimg.com/564x/d4/2c/dd/d42cddc2f24d565752724a2947c02119.jpg)


"Churn" kelimesi, müşteri kaybı anlamına gelir. Bu, bir müşterinin ürün ya da hizmet satın alma işlemi gerçekleştirdikten sonra, belirli bir zaman dilimi içerisinde tekrar satın alma işlemi yapmaması veya abonelik süresini uzatmaması anlamına gelir.

Churn oranı, müşterilerin ne sıklıkta işletmeyle tekrar etkileşimde bulunduğunu gösterir. Bu oran düşükse, müşterilerin işletmeyle olan bağlılığı yüksektir ve işletmenin kârı da artar. Ancak churn oranı yüksekse, müşteri kaybı artar ve işletmenin geliri de azalır.

Churn, özellikle abonelik bazlı hizmetlerde veya tekrarlayan satın alma işlemlerinin yapıldığı işletmelerde önemli bir metrik olarak kullanılır. İşletmeler, churn oranını azaltmak için müşterilerinin ihtiyaçlarını anlamak, onlara uygun hizmetler sunmak ve müşteri memnuniyetini artırmak gibi farklı stratejiler kullanabilirler. 

## Problem 

Telco müşteri churn verileri, üçüncü çeyrekte Kaliforniya'daki 7043 müşteriye ev telefonu ve İnternet hizmetleri sağlayan
hayali bir telekom şirketi hakkında bilgi içerir. Hangi müşterilerin hizmetlerinden ayrıldığını, kaldığını veya hizmete kaydolduğunu içermektedir.Telco şirketini terk edecek müşterileri tahmin edebilecek bir makine öğrenmesi modeli geliştirilmesi istenmektedir.

Veri seti paylaşılamamaktadır.
21 Değişken 7043 Gözlem bulunmaktadır.

## Değişkenler
- CustomerId : Müşteri İd’si
- Gender : Cinsiyet
- SeniorCitizen : Müşterinin yaşlı olup olmadığı (1, 0)
- Partner : Müşterinin bir ortağı olup olmadığı (Evet, Hayır) ? Evli olup olmama
- Dependents : Müşterinin bakmakla yükümlü olduğu kişiler olup olmadığı (Evet, Hayır) (Çocuk, anne, baba, büyükanne)
- tenure : Müşterinin şirkette kaldığı ay sayısı
- PhoneService : Müşterinin telefon hizmeti olup olmadığı (Evet, Hayır)
- MultipleLines : Müşterinin birden fazla hattı olup olmadığı (Evet, Hayır, Telefon hizmeti yok)
- InternetService : Müşterinin internet servis sağlayıcısı (DSL, Fiber optik, Hayır)
- OnlineSecurity : Müşterinin çevrimiçi güvenliğinin olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
- OnlineBackup : Müşterinin online yedeğinin olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
- DeviceProtection : Müşterinin cihaz korumasına sahip olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
- TechSupport : Müşterinin teknik destek alıp almadığı (Evet, Hayır, İnternet hizmeti yok)
- StreamingTV : Müşterinin TV yayını olup olmadığı (Evet, Hayır, İnternet hizmeti yok) Müşterinin, bir üçüncü taraf sağlayıcıdan televizyon programları yayınlamak için İnternet hizmetini kullanıp kullanmadığını gösterir
- StreamingMovies : Müşterinin film akışı olup olmadığı (Evet, Hayır, İnternet hizmeti yok) Müşterinin bir üçüncü taraf sağlayıcıdan film akışı yapmak için İnternet hizmetini kullanıp kullanmadığını gösterir
- Contract : Müşterinin sözleşme süresi (Aydan aya, Bir yıl, İki yıl)
- PaperlessBilling : Müşterinin kağıtsız faturası olup olmadığı (Evet, Hayır)
- PaymentMethod : Müşterinin ödeme yöntemi (Elektronik çek, Posta çeki, Banka havalesi (otomatik), Kredi kartı (otomatik))
- MonthlyCharges : Müşteriden aylık olarak tahsil edilen tutar
- TotalCharges : Müşteriden tahsil edilen toplam tutar
- Churn : Müşterinin kullanıp kullanmadığı (Evet veya Hayır) - Geçen ay veya çeyreklik içerisinde ayrılan müşteriler


Her satır benzersiz bir müşteriyi temsil etmekte.
Değişkenler müşteri hizmetleri, hesap ve demografik veriler hakkında bilgiler içerir.
Müşterilerin kaydolduğu hizmetler - phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies
Müşteri hesap bilgileri – ne kadar süredir müşteri oldukları, sözleşme, ödeme yöntemi, kağıtsız faturalandırma, aylık ücretler ve toplam ücretler
Müşteriler hakkında demografik bilgiler - cinsiyet, yaş aralığı ve ortakları ve bakmakla yükümlü oldukları kişiler olup olmadığı
