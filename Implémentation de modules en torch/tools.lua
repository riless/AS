print("importation des outils de génération de données et de visualisation")

function gauss(nb_pts, nb_dims, sigma, mu)
  local nb_pts = nb_pts or 20
  local nb_dims = nb_dims or 2

  local x_data=torch.randn(nb_pts, nb_dims)
  for i=1, nb_dims do
    x_data[{{}, {i}}]=x_data[{{}, {i}}]*sigma+mu[i]
  end
  return x_data
end


function generate_data(type, nb_pts, labels, sigma, mu)
  local type = type or 0

  -- generate x
  local x_data=torch.Tensor(nb_pts,#labels)
  -- mélange de 2 gaussienne
  if (type==0) then
    mus = {{-mu, -mu}, {mu, mu} }
  end
  
  -- mélange de 4 gaussienne
  if (type==1) then
    mus = {{-mu, -mu}, {mu, mu}, {-mu, mu},{mu, -mu} }
  end
  for i=1, #mus do
    l = (i-1) * (nb_pts / #mus) + 1
    k = l + (nb_pts / #mus) - 1
    g = gauss((nb_pts / #mus), #labels, sigma, mus[i])
    x_data[{{l, k}, {} }] = g
  end


  -- generate y
  local y_data=torch.Tensor(nb_pts, 1):fill(1)
  for i=1, #labels do
    a = (i-1) * (nb_pts/#labels) + 1
    b = a + (nb_pts/#labels) - 1
    y_data[{{a, b}}] = labels[i]
  end
  
  return x_data, y_data
end

-- Générer une grille 500 x 500
-- chaque point est sur deux dimensions
function grille(x,pts)
  local xmin = x:min(1)[1]
  local xmax = x:max(1)[1]
  local x1 = torch.linspace(xmin[1], xmax[1], pts)
  local x2 = torch.linspace(xmin[2], xmax[2], pts)
  local grille = torch.zeros( pts * pts, 2)
  for i=1, pts do
    for j=1, pts do
      grille[(i-1)*pts+j][1] = x1[i]
      grille[(i-1)*pts+j][2] = x2[j]
    end
  end
  return grille
end

function p(label,x, c, t)
   return {t..tostring(label), x[{{},1}], x[{{},2}], 'p ls 13 lc rgb "'..c..'"'}
end


function draw(x_data, y_data, couches, colors, bgcolors, labels, title, kernel_trick)
  
  local kernel_trick = kernel_trick or false
  local plot = {}
  -- frontière de décision
  local xGrid = grille(x_data, 500)
  if ( kernel_trick==true ) then
    xGrid = torch.cat(xGrid, torch.cmul(xGrid[{{},1}], xGrid[{{},2}]),2)
  end

  local tmpGrid = xGrid
  for i = 1, #couches-1 do
      tmpGrid = couches[i]:forward(tmpGrid)
  end
    
  local yGrid = couches[#couches]:forward(tmpGrid)
  yGrid = yGrid:sign()
  
  
  local indices = torch.linspace(1,yGrid:size(1),yGrid:size(1)):long()
  for i = 1, #labels do
    local selected = indices[yGrid:eq(labels[i])]
    if selected:nDimension(1) > 0 then
      table.insert(plot, p(labels[i], xGrid:index(1, selected), bgcolors[i], "region "))
    end
  end

  -- points
  indices = torch.linspace(1,y_data:size(1),y_data:size(1)):long()
  for i = 1, #labels do
    local selected = indices[y_data:eq(labels[i])]
    if selected:nDimension(1) > 0 then
      table.insert(plot, p(labels[i], x_data:index(1, selected), colors[i], "classe "))
    end
  end

  -- afficher la figure
  fig = gnuplot.pngfigure(title..".png")
  gnuplot.plot(plot)
  gnuplot.close()

end