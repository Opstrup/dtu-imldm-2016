#!/usr/bin/env ruby

file = open('apriori_items.txt').read

file.each_line do | line |
  supp = /Sup. \d*e\+\d*/.match(line).to_s.split(" ")[1]

  item = line.split("[")[0].gsub("_run","").gsub("_","\\_")

  puts "#{item} & #{"%.0f" % supp}\\% \\\\"
end
