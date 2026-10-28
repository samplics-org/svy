# Builds labelled-str-missing.sav with haven, the reference reader for these tests.
# Rscript labelled-str-missing.R   (haven 2.5.5)
library(haven)
s <- labelled_spss(c("a", "x", "99", "b", "zz"), na_values = c("x", "99", "zz"), label = "str var")
n <- labelled_spss(c(1, 8, 9, 2, 3), na_values = 8, na_range = c(9, 10))
write_sav(data.frame(s = s, n = n), "labelled-str-missing.sav")
