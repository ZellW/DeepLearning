cl <- h2o.init(
  max_mem_size = "3G",
  nthreads = 2)


h2oiris <- as.h2o(
  droplevels(iris[1:100, ]))

h2oiris

h2o.levels(h2oiris, 5)


write.csv(mtcars, file = "mtcars.csv")

h2omtcars <- h2o.importFile(
  path = "mtcars.csv")

h2omtcars


h2obin <- h2o.importFile(
  path = "http://www.ats.ucla.edu/stat/data/binary.csv")

h2obin
