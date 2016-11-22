#!/usr/bin/env ruby

file = open('apriori_ruless.txt').read

file.each_line do | line |
  conf = /Conf. \d*e\+\d*/.match(line).to_s.split(" ")[1]
  supp = /Sup. \d*e\+\d*/.match(line).to_s.split(" ")[1]

  impl = line.split(" <- ")[0].gsub("_","\\_")
  cond = line.split(" <- ")[1].split("[")[0].gsub("_","\\_")

  puts "#{impl} & #{cond} & #{"%.0f" % supp}\\% & #{"%.0f" % conf}\\% \\\\"
end
